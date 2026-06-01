package com.metawebthree.groupbuying.application;

import java.math.BigDecimal;
import java.sql.Timestamp;
import java.time.LocalDateTime;
import java.util.List;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.groupbuying.domain.model.GroupBuyActivityDO;
import com.metawebthree.groupbuying.domain.model.GroupBuyTeamDO;
import com.metawebthree.groupbuying.domain.model.GroupBuyOrderDO;
import com.metawebthree.groupbuying.domain.repository.GroupBuyActivityRepository;
import com.metawebthree.groupbuying.domain.repository.GroupBuyTeamRepository;
import com.metawebthree.groupbuying.domain.repository.GroupBuyOrderRepository;
import com.metawebthree.groupbuying.domain.ports.OrderPort;
import com.metawebthree.groupbuying.domain.ports.GroupBuyEventPort;

import lombok.extern.slf4j.Slf4j;

@Slf4j
@Service
public class GroupBuyApplicationService {
    private static final String STATUS_PENDING = "PENDING";
    private static final String STATUS_SUCCESS = "SUCCESS";
    private static final String STATUS_FAILED = "FAILED";
    private static final String STATUS_EXPIRED = "EXPIRED";
    private static final Integer ACTIVITY_STATUS_ACTIVE = 1;
    private static final Integer ACTIVITY_STATUS_INACTIVE = 0;

    private final GroupBuyActivityRepository activityRepository;
    private final GroupBuyTeamRepository teamRepository;
    private final GroupBuyOrderRepository orderRepository;
    private final OrderPort orderPort;
    private final GroupBuyEventPort eventPort;

    public GroupBuyApplicationService(
            GroupBuyActivityRepository activityRepository,
            GroupBuyTeamRepository teamRepository,
            GroupBuyOrderRepository orderRepository,
            OrderPort orderPort,
            GroupBuyEventPort eventPort) {
        this.activityRepository = activityRepository;
        this.teamRepository = teamRepository;
        this.orderRepository = orderRepository;
        this.orderPort = orderPort;
        this.eventPort = eventPort;
    }

    @Transactional
    public Long createActivity(String name, Long productId, String productName,
            BigDecimal singlePrice, BigDecimal groupPrice, Integer requiredQuantity,
            Integer validityHours, LocalDateTime startTime, LocalDateTime endTime) {
        Long activityId = IdWorker.getId();
        GroupBuyActivityDO activity = GroupBuyActivityDO.builder()
                .id(activityId)
                .activityName(name)
                .productId(productId)
                .productName(productName)
                .singlePrice(singlePrice)
                .groupPrice(groupPrice)
                .requiredQuantity(requiredQuantity)
                .currentQuantity(0)
                .status(ACTIVITY_STATUS_ACTIVE)
                .startTime(startTime)
                .endTime(endTime)
                .validityHours(validityHours)
                .build();
        activityRepository.save(activity);
        log.info("Group buy activity created: {}", activityId);
        return activityId;
    }

    @Transactional
    public Long createGroup(Long userId, Long activityId, String remark) {
        GroupBuyActivityDO activity = activityRepository.findById(activityId);
        validateActivity(activity);
        Long teamId = IdWorker.getId();
        String teamNo = String.valueOf(IdWorker.getId());
        LocalDateTime expireTime = LocalDateTime.now().plusHours(activity.getValidityHours());
        Timestamp expireTimestamp = Timestamp.valueOf(expireTime);
        GroupBuyTeamDO team = GroupBuyTeamDO.builder()
                .id(teamId)
                .activityId(activityId)
                .teamNo(teamNo)
                .leaderId(userId)
                .requiredQuantity(activity.getRequiredQuantity())
                .currentQuantity(1)
                .status(STATUS_PENDING)
                .expireTime(expireTimestamp)
                .build();
        teamRepository.save(team);
        Long orderId = orderPort.createOrder(userId, activity.getProductId(),
                1, activity.getGroupPrice(), remark);
        team.setOrderId(orderId);
        teamRepository.update(team);
        GroupBuyOrderDO order = GroupBuyOrderDO.builder()
                .id(IdWorker.getId())
                .teamId(teamId)
                .activityId(activityId)
                .userId(userId)
                .orderNo(String.valueOf(IdWorker.getId()))
                .orderId(orderId)
                .productId(activity.getProductId())
                .quantity(1)
                .unitPrice(activity.getGroupPrice())
                .totalAmount(activity.getGroupPrice())
                .status(STATUS_PENDING)
                .isLeader(true)
                .build();
        orderRepository.save(order);
        activity.setCurrentQuantity(activity.getCurrentQuantity() + 1);
        activityRepository.update(activity);
        eventPort.publishGroupBuyCreated(teamId, activityId, userId);
        log.info("Group buy team created: {}, leader: {}", teamId, userId);
        return teamId;
    }

    @Transactional
    public Long joinGroup(Long userId, Long teamId, String remark) {
        GroupBuyTeamDO team = teamRepository.findById(teamId);
        validateTeam(team);
        GroupBuyActivityDO activity = activityRepository.findById(team.getActivityId());
        validateActivity(activity);
        GroupBuyOrderDO existingOrder = orderRepository.findByTeamIdAndUserId(teamId, userId);
        if (existingOrder != null) {
            throw new IllegalStateException("User already in this group");
        }
        Long orderId = orderPort.createOrder(userId, activity.getProductId(),
                1, activity.getGroupPrice(), remark);
        GroupBuyOrderDO order = GroupBuyOrderDO.builder()
                .id(IdWorker.getId())
                .teamId(teamId)
                .activityId(activity.getId())
                .userId(userId)
                .orderNo(String.valueOf(IdWorker.getId()))
                .orderId(orderId)
                .productId(activity.getProductId())
                .quantity(1)
                .unitPrice(activity.getGroupPrice())
                .totalAmount(activity.getGroupPrice())
                .status(STATUS_PENDING)
                .isLeader(false)
                .build();
        orderRepository.save(order);
        team.setCurrentQuantity(team.getCurrentQuantity() + 1);
        teamRepository.update(team);
        activity.setCurrentQuantity(activity.getCurrentQuantity() + 1);
        activityRepository.update(activity);
        eventPort.publishGroupBuyJoined(teamId, userId);
        if (team.getCurrentQuantity() >= team.getRequiredQuantity()) {
            team.setStatus(STATUS_SUCCESS);
            teamRepository.update(team);
            notifyTeamSuccess(team);
            eventPort.publishGroupBuySuccess(teamId, team.getOrderId());
        }
        log.info("User {} joined group {}", userId, teamId);
        return orderId;
    }

    public List<GroupBuyActivityDO> getActiveActivities() {
        return activityRepository.findActiveActivities();
    }

    public GroupBuyTeamDO getTeamById(Long teamId) {
        return teamRepository.findById(teamId);
    }

    public List<GroupBuyOrderDO> getTeamOrders(Long teamId) {
        return orderRepository.findByTeamId(teamId);
    }

    public List<GroupBuyActivityDO> getActivitiesByStatus(Integer status) {
        return activityRepository.findByStatus(status);
    }

    private void validateActivity(GroupBuyActivityDO activity) {
        if (activity == null) {
            throw new IllegalArgumentException("Activity not found");
        }
        if (!ACTIVITY_STATUS_ACTIVE.equals(activity.getStatus())) {
            throw new IllegalStateException("Activity is not active");
        }
        LocalDateTime now = LocalDateTime.now();
        if (now.isBefore(activity.getStartTime()) || now.isAfter(activity.getEndTime())) {
            throw new IllegalStateException("Activity is not within valid time range");
        }
    }

    private void validateTeam(GroupBuyTeamDO team) {
        if (team == null) {
            throw new IllegalArgumentException("Team not found");
        }
        if (!STATUS_PENDING.equals(team.getStatus())) {
            throw new IllegalStateException("Team is not pending");
        }
        if (team.getExpireTime() != null && team.getExpireTime().before(Timestamp.valueOf(LocalDateTime.now()))) {
            team.setStatus(STATUS_EXPIRED);
            teamRepository.update(team);
            eventPort.publishGroupBuyExpired(team.getId());
            throw new IllegalStateException("Team has expired");
        }
    }

    private void notifyTeamSuccess(GroupBuyTeamDO team) {
        List<GroupBuyOrderDO> orders = orderRepository.findByTeamId(team.getId());
        for (GroupBuyOrderDO order : orders) {
            order.setStatus(STATUS_SUCCESS);
            orderRepository.update(order);
        }
        log.info("Team {} completed successfully", team.getId());
    }
}