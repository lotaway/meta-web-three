package com.metawebthree.gateway.client;

import com.metawebthree.common.generated.rpc.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import java.util.*;

@Slf4j
@Component
public class UserClient {

    @DubboReference
    private UserService userService;

    /**
     * Get user by ID
     * @param id user ID
     * @return user data map
     */
    public Map<String, Object> getUserById(String id) {
        try {
            // Use getUserPhone to get user info (simplified)
            GetUserPhoneRequest request = GetUserPhoneRequest.newBuilder()
                    .setUserId(Long.parseLong(id))
                    .build();
            GetUserPhoneResponse response = userService.getUserPhone(request);
            
            Map<String, Object> result = new HashMap<>();
            result.put("id", id);
            result.put("phone", response.getPhone());
            return result;
        } catch (Exception e) {
            log.error("Failed to get user by id: {}, error: {}", id, e.getMessage());
        }
        return new HashMap<>();
    }

    /**
     * Get user by wallet address
     * @param walletAddress wallet address
     * @return user data map
     */
    public Map<String, Object> getUserByWalletAddress(String walletAddress) {
        try {
            GetUserIdByWalletAddressRequest request = GetUserIdByWalletAddressRequest.newBuilder()
                    .setWalletAddress(walletAddress)
                    .build();
            GetUserIdByWalletAddressResponse response = userService.getUserIdByWalletAddress(request);
            
            Map<String, Object> result = new HashMap<>();
            result.put("userId", response.getUserId());
            result.put("walletAddress", walletAddress);
            return result;
        } catch (Exception e) {
            log.error("Failed to get user by wallet address: {}, error: {}", walletAddress, e.getMessage());
        }
        return new HashMap<>();
    }

    /**
     * Get wallet address by user ID
     * @param userId user ID
     * @return wallet address
     */
    public String getWalletAddressByUserId(Long userId) {
        try {
            GetWalletAddressByUserIdRequest request = GetWalletAddressByUserIdRequest.newBuilder()
                    .setUserId(userId)
                    .build();
            GetWalletAddressByUserIdResponse response = userService.getWalletAddressByUserId(request);
            return response.getWalletAddress();
        } catch (Exception e) {
            log.error("Failed to get wallet address by userId: {}, error: {}", userId, e.getMessage());
        }
        return "";
    }

    /**
     * Get user integration points
     * @param userId user ID
     * @return integration points
     */
    public long getUserIntegration(Long userId) {
        try {
            GetUserIntegrationRequest request = GetUserIntegrationRequest.newBuilder()
                    .setUserId(userId)
                    .build();
            GetUserIntegrationResponse response = userService.getUserIntegration(request);
            return response.getIntegration();
        } catch (Exception e) {
            log.error("Failed to get user integration: userId={}, error: {}", userId, e.getMessage());
        }
        return 0;
    }

    /**
     * Add integration points to user
     * @param userId user ID
     * @param amount amount to add
     * @return true if success
     */
    public boolean addIntegration(Long userId, long amount) {
        try {
            AddIntegrationRequest request = AddIntegrationRequest.newBuilder()
                    .setUserId(userId)
                    .setIntegration((int) amount)
                    .build();
            AddIntegrationResponse response = userService.addIntegration(request);
            return response.getSuccess();
        } catch (Exception e) {
            log.error("Failed to add integration: userId={}, amount={}, error: {}", userId, amount, e.getMessage());
        }
        return false;
    }

    /**
     * Add growth value to user
     * @param userId user ID
     * @param amount amount to add
     * @return true if success
     */
    public boolean addGrowth(Long userId, long amount) {
        try {
            AddGrowthRequest request = AddGrowthRequest.newBuilder()
                    .setUserId(userId)
                    .setGrowth((int) amount)
                    .build();
            AddGrowthResponse response = userService.addGrowth(request);
            return response.getSuccess();
        } catch (Exception e) {
            log.error("Failed to add growth: userId={}, amount={}, error: {}", userId, amount, e.getMessage());
        }
        return false;
    }

    /**
     * Get users with pagination
     * @param page page number
     * @param size page size
     * @return users connection map
     */
    public Map<String, Object> getUsers(Integer page, Integer size) {
        try {
            // Create empty connection for now - requires a list users RPC
            Map<String, Object> connection = new HashMap<>();
            connection.put("edges", new ArrayList<>());
            connection.put("totalCount", 0);
            connection.put("pageInfo", Map.of(
                "hasNextPage", false,
                "hasPreviousPage", page != null && page > 0
            ));
            return connection;
        } catch (Exception e) {
            log.error("Failed to get users: page={}, size={}, error: {}", page, size, e.getMessage());
        }
        Map<String, Object> connection = new HashMap<>();
        connection.put("edges", new ArrayList<>());
        connection.put("totalCount", 0);
        connection.put("pageInfo", Map.of(
            "hasNextPage", false,
            "hasPreviousPage", page != null && page > 0
        ));
        return connection;
    }
}