package com.metawebthree.promotion.infrastructure.gateway;

import com.metawebthree.common.generated.rpc.GetWalletAddressByUserIdRequest;
import com.metawebthree.common.generated.rpc.GetWalletAddressByUserIdResponse;
import com.metawebthree.common.generated.rpc.UserService;
import com.metawebthree.promotion.domain.ports.UserWalletService;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Service;

@Service
public class UserWalletServiceRPCImpl implements UserWalletService {

    @DubboReference
    private UserService userService;

    @Override
    public String getWalletAddressByUserId(Long userId) {
        if (userId == null) return null;
        GetWalletAddressByUserIdRequest request = GetWalletAddressByUserIdRequest.newBuilder()
                .setUserId(userId)
                .build();
        GetWalletAddressByUserIdResponse response = userService.getWalletAddressByUserId(request);
        String wallet = response.getWalletAddress();
        return (wallet != null && !wallet.isEmpty()) ? wallet : null;
    }
}
