package com.metawebthree.user.infrastructure.rpc;

import com.metawebthree.common.generated.rpc.*;
import com.metawebthree.user.application.dto.UserDTO;
import com.metawebthree.user.infrastructure.persistence.mapper.UserMapper;
import org.apache.dubbo.config.annotation.DubboService;
import org.springframework.stereotype.Service;
import java.util.concurrent.CompletableFuture;

@DubboService
@Service
public class UserRPCServiceImpl implements UserService {
    private final UserMapper userMapper;

    public UserRPCServiceImpl(UserMapper userMapper) {
        this.userMapper = userMapper;
    }

    @Override
    public GetWalletAddressByUserIdResponse getWalletAddressByUserId(GetWalletAddressByUserIdRequest request) {
        UserDTO user = userMapper.findByIdWithWallet(request.getUserId());
        String wallet = (user != null && user.getWalletAddress() != null) ? user.getWalletAddress() : "";
        return GetWalletAddressByUserIdResponse.newBuilder().setWalletAddress(wallet).build();
    }

    @Override
    public CompletableFuture<GetWalletAddressByUserIdResponse> getWalletAddressByUserIdAsync(GetWalletAddressByUserIdRequest request) {
        return CompletableFuture.completedFuture(getWalletAddressByUserId(request));
    }

    @Override
    public GetUserIdByWalletAddressResponse getUserIdByWalletAddress(GetUserIdByWalletAddressRequest request) {
        UserDTO user = userMapper.findByWalletAddress(request.getWalletAddress());
        long userId = (user != null) ? user.getId() : 0L;
        return GetUserIdByWalletAddressResponse.newBuilder().setUserId(userId).build();
    }

    @Override
    public CompletableFuture<GetUserIdByWalletAddressResponse> getUserIdByWalletAddressAsync(GetUserIdByWalletAddressRequest request) {
        return CompletableFuture.completedFuture(getUserIdByWalletAddress(request));
    }
}
