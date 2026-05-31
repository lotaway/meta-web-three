package com.metawebthree.user.infrastructure.rpc;

import com.metawebthree.common.generated.rpc.*;
import com.metawebthree.user.application.dto.UserDTO;
import com.metawebthree.user.infrastructure.persistence.mapper.UserMapper;
import org.apache.dubbo.config.annotation.DubboService;
import org.springframework.stereotype.Service;
import lombok.extern.slf4j.Slf4j;

import java.util.concurrent.CompletableFuture;

@Slf4j
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

    @Override
    public ReturnIntegrationResponse returnIntegration(ReturnIntegrationRequest request) {
        ReturnIntegrationResponse.Builder responseBuilder = ReturnIntegrationResponse.newBuilder();
        try {
            if (request.getIntegration() == null || request.getIntegration() <= 0) {
                responseBuilder.setSuccess(false).setMessage("积分数量必须大于0");
                return responseBuilder.build();
            }
            
            int updated = userMapper.updateIntegration(request.getUserId(), request.getIntegration());
            if (updated > 0) {
                Integer currentIntegration = userMapper.getIntegration(request.getUserId());
                responseBuilder.setSuccess(true)
                        .setMessage("积分已返还")
                        .setCurrentIntegration(currentIntegration != null ? currentIntegration : 0);
                log.info("积分返还成功 - userId: {}, orderId: {}, integration: {}", 
                        request.getUserId(), request.getOrderId(), request.getIntegration());
            } else {
                responseBuilder.setSuccess(false).setMessage("用户不存在或积分返还失败");
            }
        } catch (Exception e) {
            log.error("积分返还失败 - userId: {}, orderId: {}, error: {}", 
                    request.getUserId(), request.getOrderId(), e.getMessage(), e);
            responseBuilder.setSuccess(false).setMessage("积分返还失败: " + e.getMessage());
        }
        return responseBuilder.build();
    }

    @Override
    public GetUserIntegrationResponse getUserIntegration(GetUserIntegrationRequest request) {
        GetUserIntegrationResponse.Builder responseBuilder = GetUserIntegrationResponse.newBuilder();
        try {
            Integer integration = userMapper.getIntegration(request.getUserId());
            responseBuilder.setUserId(request.getUserId())
                    .setIntegration(integration != null ? integration : 0);
        } catch (Exception e) {
            log.error("获取用户积分失败 - userId: {}, error: {}", request.getUserId(), e.getMessage(), e);
            responseBuilder.setUserId(request.getUserId()).setIntegration(0);
        }
        return responseBuilder.build();
    }
}
