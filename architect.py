""" Architect controls architecture of cell by computing gradients of alphas 
    NAS训练算法中的第1步: 更新架构参数 α
    根据论文可知 dα Lval(w*, α) 约等于 dα Lval(w', α)    w' = w - ξ * dw Ltrain(w, α)
"""
import copy
import torch
import torch.nn.functional as F


class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net                      # network
        self.v_net = copy.deepcopy(net)     # 不直接用外面的optimizer来进行w的更新，而是自己新建一个network，主要是因为我们这里的更新不能对Network的w进行更新
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay    # 正则化项用来防止过拟合

    def virtual_step(self, train_input1,train_input2, xi, w_optim,labels,train_idx):
        """
        Compute unrolled weight w' (virtual step)

        根据公式计算 w' = w - ξ * dw Ltrain(w, α)   
        Monmentum公式：  dw Ltrain -> v * w_momentum + dw Ltrain + w_weight_decay * w 
        -> m + g + 正则项
  
        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)  即公式中的 ξ
            w_optim: weights optimizer 用来更新 w 的优化器
        """
        # forward & calc loss
        logits,_,_ = self.net(train_input1, train_input2) # L_trn(w)
        logp = F.log_softmax(logits, 1)
        train_loss = F.nll_loss(logp, labels[train_idx])
        # compute gradient 计算  dw L_trn(w) = g
        gradients = torch.autograd.grad(train_loss, self.net.w_parameter())

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.net.w_parameter(), self.v_net.w_parameter(), gradients):
                # m = v * w_momentum  用的就是Network进行w更新的momentum
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum

                # 做一步momentum梯度下降后更新得到 w' = w - ξ * (m + dw Ltrain(w, α) + 正则项 )
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w))   

            # synchronize alphas 更新了v_net的alpha
            for comp, vcomp in zip(self.net.comp(), self.v_net.comp()):
                vcomp.copy_(comp)

    def unrolled_backward(self, tr_input1, tr_input2, val_input1, val_input2, xi, w_optim, labels, train_idx_batch
                          , val_idx_batch):
        """ Compute unrolled loss and backward its gradients
        计算目标函数关于 α 的近似梯度
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(tr_input1, tr_input2, xi, w_optim,labels,train_idx_batch)

        # calc unrolled loss
        logits, _, _ = self.v_net(val_input1, val_input2) # L_val(w', α)  在使用w', 新alpha的net上计算损失值
        logp = F.log_softmax(logits, 1)
        val_loss = F.nll_loss(logp, labels[val_idx_batch])

        # compute gradient
        v_comp = tuple(self.v_net.comp())
        v_w = tuple(self.v_net.w_parameter())
        v_grads = torch.autograd.grad(val_loss, v_comp + v_w)
        dcomp = v_grads[:len(v_comp)]    # dα L_val(w', α)   梯度近似后公式第一项
        dw = v_grads[len(v_comp):]        # dw' L_val(w', α)  梯度近似后公式第二项的第二个乘数

        hessian = self.compute_hessian(dw, tr_input1, tr_input2, labels, train_idx_batch)        # 梯度近似后公式第二项

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for comp, dcomp, h in zip(self.net.comp(), dcomp, hessian):
                comp.grad = dcomp - xi*h    # 求出了目标函数的近似梯度值

    def unrolled_backward_full_batch(self, tr_input1, tr_input2,tr_input3, val_input1, val_input2, val_input3, xi, w_optim, labels,train_idx,val_idx):
        """ Compute unrolled loss and backward its gradients
        计算目标函数关于 α 的近似梯度
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step_full_batch(tr_input1, tr_input2, tr_input3, xi, w_optim,labels,train_idx)

        # calc unrolled loss
        logits, _, _, _ = self.v_net(val_input1, val_input2,val_input3) # L_val(w', α)  在使用w', 新alpha的net上计算损失值
        logp = F.log_softmax(logits, 1)
        val_loss = F.nll_loss(logp[val_idx], labels[val_idx])

        # compute gradient
        v_comp = tuple(self.v_net.comp())
        v_w = tuple(self.v_net.w_parameter())
        v_grads = torch.autograd.grad(val_loss, v_comp + v_w)
        dcomp = v_grads[:len(v_comp)]    # dα L_val(w', α)   梯度近似后公式第一项
        dw = v_grads[len(v_comp):]        # dw' L_val(w', α)  梯度近似后公式第二项的第二个乘数

        hessian = self.compute_hessian_full_batch(dw, tr_input1, tr_input2, tr_input3,labels, train_idx)        # 梯度近似后公式第二项

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for comp, dcomp, h in zip(self.net.comp(), dcomp, hessian):
                comp.grad = dcomp - xi*h    # 求出了目标函数的近似梯度值

    def virtual_step_full_batch(self, train_input1, train_input2, train_input3, xi, w_optim, labels, train_idx):
        """
        Compute unrolled weight w' (virtual step)

        根据公式计算 w' = w - ξ * dw Ltrain(w, α)
        Monmentum公式：  dw Ltrain -> v * w_momentum + dw Ltrain + w_weight_decay * w
        -> m + g + 正则项

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)  即公式中的 ξ
            w_optim: weights optimizer 用来更新 w 的优化器
        """
        # forward & calc loss
        logits, _, _, _ = self.net(train_input1, train_input2, train_input3)  # L_trn(w)
        logp = F.log_softmax(logits, 1)
        train_loss = F.nll_loss(logp[train_idx], labels[train_idx])
        # compute gradient 计算  dw L_trn(w) = g
        gradients = torch.autograd.grad(train_loss, self.net.w_parameter())

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.net.w_parameter(), self.v_net.w_parameter(), gradients):
                # m = v * w_momentum  用的就是Network进行w更新的momentum
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum

                # 做一步momentum梯度下降后更新得到 w' = w - ξ * (m + dw Ltrain(w, α) + 正则项 )
                vw.copy_(w - xi * (m + g + self.w_weight_decay * w))

                # synchronize alphas 更新了v_net的alpha
            for comp, vcomp in zip(self.net.comp(), self.v_net.comp()):
                vcomp.copy_(comp)

    def compute_hessian(self, dw, tr_input1, tr_input2, labels,train_idx_batch):
        """
        求经过泰勒展开后的第二项的近似值
        dw = dw` { L_val(w`, alpha) }  输入里已经给了所有预测数据的dw
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)    [1]
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()   # 把每个 w 先拉成一行，然后把所有的 w 摞起来，变成 n 行, 然后求L2值
        eps = 0.01 / norm

        # w+ = w + eps * dw`
        with torch.no_grad():
            for p, d in zip(self.net.w_parameter(), dw):
                p += eps * d        # 将model中所有的w'更新成 w+
        logits, _, _ = self.net(tr_input1, tr_input2)  # L_trn(w)
        logp = F.log_softmax(logits, 1)
        train_loss = F.nll_loss(logp, labels[train_idx_batch])     # L_trn(w+)
        dcomp_pos = torch.autograd.grad(train_loss, self.net.comp()) # dalpha { L_trn(w+) }

        # w- = w - eps * dw`
        with torch.no_grad():
            for p, d in zip(self.net.w_parameter(), dw):
                p -= 2. * eps * d   # 将model中所有的w'更新成 w-,   w- = w - eps * dw = w+ - eps * dw * 2, 现在的 p 是 w+
        logits, _, _ = self.net(tr_input1, tr_input2)  # L_trn(w)
        logp = F.log_softmax(logits, 1)
        train_loss = F.nll_loss(logp, labels[train_idx_batch])  # L_trn(w-)
        dcomp_neg = torch.autograd.grad(train_loss, self.net.comp()) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.w_parameter(), dw):
                p += eps * d        # 将模型的参数从 w- 恢复成 w,  w = w- + eps * dw

        hessian = [(p-n) / 2.*eps for p, n in zip(dcomp_pos, dcomp_neg)]  # 利用公式 [1] 计算泰勒展开后第二项的近似值返回
        return hessian
    def compute_hessian_full_batch(self, dw, tr_input1, tr_input2,tr_input3, labels,train_idx_batch):
        """
        求经过泰勒展开后的第二项的近似值
        dw = dw` { L_val(w`, alpha) }  输入里已经给了所有预测数据的dw
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)    [1]
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()   # 把每个 w 先拉成一行，然后把所有的 w 摞起来，变成 n 行, 然后求L2值
        eps = 0.01 / norm

        # w+ = w + eps * dw`
        with torch.no_grad():
            for p, d in zip(self.net.w_parameter(), dw):
                p += eps * d        # 将model中所有的w'更新成 w+
        logits, _, _, _ = self.net(tr_input1, tr_input2,tr_input3)  # L_trn(w)
        logp = F.log_softmax(logits, 1)
        train_loss = F.nll_loss(logp[train_idx_batch], labels[train_idx_batch])     # L_trn(w+)
        dcomp_pos = torch.autograd.grad(train_loss, self.net.comp()) # dalpha { L_trn(w+) }

        # w- = w - eps * dw`
        with torch.no_grad():
            for p, d in zip(self.net.w_parameter(), dw):
                p -= 2. * eps * d   # 将model中所有的w'更新成 w-,   w- = w - eps * dw = w+ - eps * dw * 2, 现在的 p 是 w+
        logits, _, _, _ = self.net(tr_input1, tr_input2,tr_input3)  # L_trn(w)
        logp = F.log_softmax(logits, 1)
        train_loss = F.nll_loss(logp[train_idx_batch], labels[train_idx_batch])  # L_trn(w-)
        dcomp_neg = torch.autograd.grad(train_loss, self.net.comp()) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.w_parameter(), dw):
                p += eps * d        # 将模型的参数从 w- 恢复成 w,  w = w- + eps * dw

        hessian = [(p-n) / 2.*eps for p, n in zip(dcomp_pos, dcomp_neg)]  # 利用公式 [1] 计算泰勒展开后第二项的近似值返回
        return hessian
