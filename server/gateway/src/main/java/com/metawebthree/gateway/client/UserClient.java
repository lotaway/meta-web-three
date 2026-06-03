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

    public Map<String, Object> getUserById(String id) {
        try {
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

    public Map<String, Object> getUsers(Integer page, Integer size) {
        try {
            ListUsersRequest request = ListUsersRequest.newBuilder()
                    .setPage(page != null ? page : 0)
                    .setSize(size != null ? size : 20)
                    .build();
            ListUsersResponse response = userService.listUsers(request);

            return buildUsersConnection(
                buildUserEdges(response.getUsersList()),
                response.getTotalCount(),
                response.getPage(),
                response.getSize()
            );
        } catch (Exception e) {
            log.error("Failed to get users: page={}, size={}, error: {}", page, size, e.getMessage());
        }
        return createEmptyUsersConnection(page);
    }

    private List<Map<String, Object>> buildUserEdges(List<UserInfoProto> users) {
        List<Map<String, Object>> edges = new ArrayList<>();
        for (UserInfoProto user : users) {
            Map<String, Object> node = new HashMap<>();
            node.put("id", user.getId());
            node.put("username", user.getUsername());
            node.put("phone", user.getPhone());
            node.put("email", user.getEmail());
            node.put("avatar", user.getAvatar());
            node.put("status", user.getStatus());
            node.put("createdAt", user.getCreatedAt());
            Map<String, Object> edge = new HashMap<>();
            edge.put("node", node);
            edges.add(edge);
        }
        return edges;
    }

    private Map<String, Object> buildUsersConnection(List<Map<String, Object>> edges, int totalCount, int currentPage, int pageSize) {
        Map<String, Object> connection = new HashMap<>();
        connection.put("edges", edges);
        connection.put("totalCount", totalCount);
        connection.put("pageInfo", Map.of(
            "hasNextPage", (currentPage + 1) * pageSize < totalCount,
            "hasPreviousPage", currentPage > 0
        ));
        return connection;
    }

    private Map<String, Object> createEmptyUsersConnection(Integer page) {
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
