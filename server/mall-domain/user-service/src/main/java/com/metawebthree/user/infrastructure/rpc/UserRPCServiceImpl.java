package com.metawebthree.user.infrastructure.rpc;

import com.metawebthree.common.generated.rpc.*;
import com.metawebthree.user.application.dto.UserDTO;
import com.metawebthree.user.application.MemberLevelService;
import com.metawebthree.user.domain.model.MemberLevelDO;
import com.metawebthree.user.domain.model.UserDO;
import com.metawebthree.user.infrastructure.persistence.mapper.UserMapper;
import org.apache.dubbo.config.annotation.DubboService;
import org.springframework.stereotype.Service;
import lombok.extern.slf4j.Slf4j;

import java.util.Comparator;
import java.util.List;
import java.util.concurrent.CompletableFuture;

@Slf4j
@DubboService
@Service
public class UserRPCServiceImpl implements UserService {
    private final UserMapper userMapper;
    private final MemberLevelService memberLevelService;

    public UserRPCServiceImpl(UserMapper userMapper, MemberLevelService memberLevelService) {
        this.userMapper = userMapper;
        this.memberLevelService = memberLevelService;
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
            if (request.getIntegration() <= 0) {
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
    public CompletableFuture<ReturnIntegrationResponse> returnIntegrationAsync(ReturnIntegrationRequest request) {
        return CompletableFuture.completedFuture(returnIntegration(request));
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

    @Override
    public CompletableFuture<GetUserIntegrationResponse> getUserIntegrationAsync(GetUserIntegrationRequest request) {
        return CompletableFuture.completedFuture(getUserIntegration(request));
    }

    @Override
    public AddIntegrationResponse addIntegration(AddIntegrationRequest request) {
        AddIntegrationResponse.Builder responseBuilder = AddIntegrationResponse.newBuilder();
        try {
            if (request.getIntegration() <= 0) {
                responseBuilder.setSuccess(false).setMessage("积分数量必须大于0");
                return responseBuilder.build();
            }
            
            int updated = userMapper.updateIntegration(request.getUserId(), request.getIntegration());
            if (updated > 0) {
                Integer currentIntegration = userMapper.getIntegration(request.getUserId());
                responseBuilder.setSuccess(true)
                        .setMessage("积分已添加")
                        .setCurrentIntegration(currentIntegration != null ? currentIntegration : 0);
                log.info("积分添加成功 - userId: {}, orderId: {}, integration: {}", 
                        request.getUserId(), request.getOrderId(), request.getIntegration());
            } else {
                responseBuilder.setSuccess(false).setMessage("用户不存在或积分添加失败");
            }
        } catch (Exception e) {
            log.error("积分添加失败 - userId: {}, orderId: {}, error: {}", 
                    request.getUserId(), request.getOrderId(), e.getMessage(), e);
            responseBuilder.setSuccess(false).setMessage("积分添加失败: " + e.getMessage());
        }
        return responseBuilder.build();
    }

    @Override
    public CompletableFuture<AddIntegrationResponse> addIntegrationAsync(AddIntegrationRequest request) {
        return CompletableFuture.completedFuture(addIntegration(request));
    }

    @Override
    public AddGrowthResponse addGrowth(AddGrowthRequest request) {
        AddGrowthResponse.Builder responseBuilder = AddGrowthResponse.newBuilder();
        try {
            if (request.getGrowth() <= 0) {
                responseBuilder.setSuccess(false).setMessage("成长值必须大于0");
                return responseBuilder.build();
            }
            
            int updated = userMapper.updateGrowth(request.getUserId(), request.getGrowth());
            if (updated > 0) {
                Integer currentGrowth = userMapper.getGrowth(request.getUserId());
                responseBuilder.setSuccess(true)
                        .setMessage("成长值已添加")
                        .setCurrentGrowth(currentGrowth != null ? currentGrowth : 0);
                log.info("成长值添加成功 - userId: {}, orderId: {}, growth: {}", 
                        request.getUserId(), request.getOrderId(), request.getGrowth());
                
                updateMemberLevelIfNeeded(request.getUserId(), currentGrowth);
            } else {
                responseBuilder.setSuccess(false).setMessage("用户不存在或成长值添加失败");
            }
        } catch (Exception e) {
            log.error("成长值添加失败 - userId: {}, orderId: {}, error: {}", 
                    request.getUserId(), request.getOrderId(), e.getMessage(), e);
            responseBuilder.setSuccess(false).setMessage("成长值添加失败: " + e.getMessage());
        }
        return responseBuilder.build();
    }

    private void updateMemberLevelIfNeeded(Long userId, Integer currentGrowth) {
        if (currentGrowth == null || currentGrowth <= 0) {
            return;
        }
        
        List<MemberLevelDO> allLevels = memberLevelService.list();
        if (allLevels == null || allLevels.isEmpty()) {
            return;
        }
        
        MemberLevelDO targetLevel = allLevels.stream()
                .filter(level -> level.getGrowthPoint() != null && level.getGrowthPoint() <= currentGrowth)
                .max(Comparator.comparing(MemberLevelDO::getGrowthPoint))
                .orElse(null);
        
        if (targetLevel == null) {
            return;
        }
        
        Long currentMemberLevelId = userMapper.getMemberLevelId(userId);
        
        if (currentMemberLevelId == null || !currentMemberLevelId.equals(targetLevel.getId())) {
            userMapper.updateMemberLevelId(userId, targetLevel.getId());
            log.info("会员等级自动升级 - userId: {}, oldLevelId: {}, newLevelId: {}, newLevelName: {}, growth: {}", 
                    userId, currentMemberLevelId, targetLevel.getId(), targetLevel.getName(), currentGrowth);
        }
    }

    @Override
    public CompletableFuture<AddGrowthResponse> addGrowthAsync(AddGrowthRequest request) {
        return CompletableFuture.completedFuture(addGrowth(request));
    }
    
    @Override
    public GetUserPhoneResponse getUserPhone(GetUserPhoneRequest request) {
        GetUserPhoneResponse.Builder responseBuilder = GetUserPhoneResponse.newBuilder();
        try {
            String phone = userMapper.selectTelephoneByUserId(request.getUserId());
            if (phone != null && !phone.isEmpty()) {
                responseBuilder.setSuccess(true)
                        .setPhone(phone)
                        .setMessage("查询成功");
                log.info("获取用户手机号成功 - userId: {}, phone: ***", request.getUserId());
            } else {
                responseBuilder.setSuccess(false)
                        .setPhone("")
                        .setMessage("用户未绑定手机号");
                log.warn("用户未绑定手机号 - userId: {}", request.getUserId());
            }
        } catch (Exception e) {
            log.error("获取用户手机号失败 - userId: {}, error: {}", request.getUserId(), e.getMessage(), e);
            responseBuilder.setSuccess(false)
                    .setPhone("")
                    .setMessage("查询失败: " + e.getMessage());
        }
        return responseBuilder.build();
    }
    
    @Override
    public CompletableFuture<GetUserPhoneResponse> getUserPhoneAsync(GetUserPhoneRequest request) {
        return CompletableFuture.completedFuture(getUserPhone(request));
    }

    @Override
    public ListUsersResponse listUsers(ListUsersRequest request) {
        log.info("Dubbo RPC: listUsers called with page: {}, size: {}", request.getPage(), request.getSize());
        try {
            int page = request.getPage() > 0 ? request.getPage() : 1;
            int size = request.getSize() > 0 ? request.getSize() : 10;

            com.baomidou.mybatisplus.core.metadata.IPage<UserDO> pageResult = userMapper.selectPage(
                    new com.baomidou.mybatisplus.extension.plugins.pagination.Page<>(page, size),
                    new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<UserDO>()
                            .orderByDesc(UserDO::getId));

            List<UserInfoProto> users = pageResult.getRecords().stream()
                    .map(u -> UserInfoProto.newBuilder()
                            .setId(u.getId() != null ? u.getId() : 0L)
                            .setUsername(u.getUsername() != null ? u.getUsername() : "")
                            .setPhone(u.getTelephone() != null ? u.getTelephone() : "")
                            .setAvatar(u.getAvatar() != null ? u.getAvatar() : "")
                            .build())
                    .toList();

            return ListUsersResponse.newBuilder()
                    .addAllUsers(users)
                    .setTotalCount((int) pageResult.getTotal())
                    .setPage(page)
                    .setSize(size)
                    .build();
        } catch (Exception e) {
            log.error("Failed to list users", e);
            return ListUsersResponse.newBuilder().build();
        }
    }

    @Override
    public CompletableFuture<ListUsersResponse> listUsersAsync(ListUsersRequest request) {
        return CompletableFuture.completedFuture(listUsers(request));
    }
}
