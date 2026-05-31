package com.metawebthree.order.infrastructure.client;

import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import com.metawebthree.common.generated.rpc.GetUserIntegrationRequest;
import com.metawebthree.common.generated.rpc.GetUserIntegrationResponse;
import com.metawebthree.common.generated.rpc.ReturnIntegrationRequest;
import com.metawebthree.common.generated.rpc.ReturnIntegrationResponse;
import com.metawebthree.common.generated.rpc.UserService;

import lombok.extern.slf4j.Slf4j;

@Component
@Slf4j
public class UserClient {

    @DubboReference(check = false, lazy = true)
    private UserService userService;

    /**
     * 返还用户使用的积分
     * @param userId 用户ID
     * @param integration 积分数量
     * @param orderId 订单ID
     * @return 返还后的当前积分，null表示失败
     */
    public Integer returnIntegration(Long userId, int integration, Long orderId) {
        try {
            ReturnIntegrationRequest request = ReturnIntegrationRequest.newBuilder()
                    .setUserId(userId)
                    .setIntegration(integration)
                    .setOrderId(orderId)
                    .setReason("订单取消返还积分")
                    .build();

            ReturnIntegrationResponse response = userService.returnIntegration(request);
            if (response.getSuccess()) {
                log.info("积分返还成功 - userId: {}, orderId: {}, integration: {}, current: {}", 
                        userId, orderId, integration, response.getCurrentIntegration());
                return response.getCurrentIntegration();
            } else {
                log.warn("积分返还失败 - userId: {}, orderId: {}, message: {}", 
                        userId, orderId, response.getMessage());
                return null;
            }
        } catch (Exception e) {
            log.error("积分返还异常 - userId: {}, orderId: {}, error: {}", 
                    userId, orderId, e.getMessage(), e);
            return null;
        }
    }

    /**
     * 获取用户当前积分
     * @param userId 用户ID
     * @return 当前积分，null表示失败
     */
    public Integer getUserIntegration(Long userId) {
        try {
            GetUserIntegrationRequest request = GetUserIntegrationRequest.newBuilder()
                    .setUserId(userId)
                    .build();

            GetUserIntegrationResponse response = userService.getUserIntegration(request);
            return response.getIntegration();
        } catch (Exception e) {
            log.error("获取用户积分失败 - userId: {}, error: {}", userId, e.getMessage(), e);
            return null;
        }
    }
}