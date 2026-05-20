package com.metawebthree.user.domain.ports;

public interface ReferralBindingPort {
    void bind(Long userId, Long referrerId);
}
