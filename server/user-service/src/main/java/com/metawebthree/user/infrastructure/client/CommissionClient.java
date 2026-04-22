package com.metawebthree.user.infrastructure.client;

import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import com.metawebthree.common.generated.rpc.BindRequest;
import com.metawebthree.common.generated.rpc.CommissionService;
import com.metawebthree.user.domain.ports.ReferralBindingPort;

@Component
public class CommissionClient implements ReferralBindingPort {

    @DubboReference
    private CommissionService commissionService;

    @Override
    public void bind(Long userId, Long referrerId) {
        validate(userId, referrerId);
        
        BindRequest request = BindRequest.newBuilder()
                .setUserId(userId)
                .setParentUserId(referrerId)
                .build();
        
        commissionService.bind(request);
    }

    private void validate(Long userId, Long referrerId) {
        if (userId == null || referrerId == null || referrerId <= 0) {
            throw new IllegalArgumentException("invalid referral binding");
        }
    }
}
