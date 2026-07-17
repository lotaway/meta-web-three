package com.metawebthree.crm.adapter.client;

import com.metawebthree.common.generated.rpc.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import java.util.*;

@Slf4j
@Component
public class UserServiceClient {

    @DubboReference(check = false, lazy = true)
    private UserService userService;

    public Map<String, Object> getUserById(Long userId) {
        try {
            GetUserPhoneRequest request = GetUserPhoneRequest.newBuilder()
                    .setUserId(userId)
                    .build();
            GetUserPhoneResponse response = userService.getUserPhone(request);

            Map<String, Object> result = new HashMap<>();
            result.put("id", userId);
            result.put("phone", response.getPhone());
            return result;
        } catch (Exception e) {
            log.error("Failed to get user by id: {}, error: {}", userId, e.getMessage());
            throw new RuntimeException("Failed to get user by id: " + userId, e);
        }
    }

    public List<Map<String, Object>> searchUsers(String keyword) {
        try {
            ListUsersRequest request = ListUsersRequest.newBuilder()
                    .setPage(0)
                    .setSize(100)
                    .build();
            ListUsersResponse response = userService.listUsers(request);

            List<Map<String, Object>> users = new ArrayList<>();
            for (UserInfoProto proto : response.getUsersList()) {
                Map<String, Object> user = new HashMap<>();
                user.put("id", proto.getId());
                user.put("username", proto.getUsername());
                user.put("phone", proto.getPhone());
                user.put("email", proto.getEmail());
                user.put("avatar", proto.getAvatar());
                user.put("status", proto.getStatus());
                user.put("createdAt", proto.getCreatedAt());
                users.add(user);
            }
            return users;
        } catch (Exception e) {
            log.error("Failed to search users, error: {}", e.getMessage());
            throw new RuntimeException("Failed to search users", e);
        }
    }

    public Map<String, Object> getUserStatistics() {
        try {
            ListUsersRequest request = ListUsersRequest.newBuilder()
                    .setPage(0)
                    .setSize(1)
                    .build();
            ListUsersResponse response = userService.listUsers(request);

            Map<String, Object> stats = new HashMap<>();
            stats.put("totalUsers", response.getTotalCount());
            return stats;
        } catch (Exception e) {
            log.error("Failed to get user statistics, error: {}", e.getMessage());
            throw new RuntimeException("Failed to get user statistics", e);
        }
    }
}
