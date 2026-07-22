package com.metawebthree.crm.adapter.client;

import com.metawebthree.common.generated.rpc.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

@Slf4j
@Component
public class UserServiceClient {

    public record UserDTO(Long id, String username, String phone, String email, String avatar, Integer status, Long createdAt) {}

    public record UserStatsDTO(long totalUsers) {}

    @DubboReference(check = false, lazy = true)
    private UserService userService;

    public UserDTO getUserById(Long userId) {
        try {
            GetUserPhoneRequest request = GetUserPhoneRequest.newBuilder()
                    .setUserId(userId)
                    .build();
            GetUserPhoneResponse response = userService.getUserPhone(request);

            return new UserDTO(userId, null, response.getPhone(), null, null, null, null);
        } catch (Exception e) {
            log.error("Failed to get user by id: {}, error: {}", userId, e.getMessage());
            throw new RuntimeException("Failed to get user by id: " + userId, e);
        }
    }

    public List<UserDTO> searchUsers(String keyword) {
        try {
            ListUsersRequest request = ListUsersRequest.newBuilder()
                    .setPage(0)
                    .setSize(100)
                    .build();
            ListUsersResponse response = userService.listUsers(request);

            List<UserDTO> users = new ArrayList<>();
            for (UserInfoProto proto : response.getUsersList()) {
                users.add(new UserDTO(proto.getId(), proto.getUsername(), proto.getPhone(), proto.getEmail(), proto.getAvatar(), proto.getStatus(), proto.getCreatedAt()));
            }
            return users;
        } catch (Exception e) {
            log.error("Failed to search users, error: {}", e.getMessage());
            throw new RuntimeException("Failed to search users", e);
        }
    }

    public UserStatsDTO getUserStatistics() {
        try {
            ListUsersRequest request = ListUsersRequest.newBuilder()
                    .setPage(0)
                    .setSize(1)
                    .build();
            ListUsersResponse response = userService.listUsers(request);

            return new UserStatsDTO(response.getTotalCount());
        } catch (Exception e) {
            log.error("Failed to get user statistics, error: {}", e.getMessage());
            throw new RuntimeException("Failed to get user statistics", e);
        }
    }
}
