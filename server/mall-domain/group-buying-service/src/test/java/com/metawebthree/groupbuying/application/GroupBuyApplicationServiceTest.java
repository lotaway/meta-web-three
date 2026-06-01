package com.metawebthree.groupbuying.application;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.metawebthree.groupbuying.domain.model.GroupBuyActivityDO;
import com.metawebthree.groupbuying.domain.model.GroupBuyTeamDO;
import com.metawebthree.groupbuying.domain.model.GroupBuyOrderDO;
import com.metawebthree.groupbuying.domain.repository.GroupBuyActivityRepository;
import com.metawebthree.groupbuying.domain.repository.GroupBuyTeamRepository;
import com.metawebthree.groupbuying.domain.repository.GroupBuyOrderRepository;
import com.metawebthree.groupbuying.domain.ports.OrderPort;
import com.metawebthree.groupbuying.domain.ports.GroupBuyEventPort;

class GroupBuyApplicationServiceTest {

    private GroupBuyActivityRepository activityRepository;
    private GroupBuyTeamRepository teamRepository;
    private GroupBuyOrderRepository orderRepository;
    private OrderPort orderPort;
    private GroupBuyEventPort eventPort;
    private GroupBuyApplicationService service;

    @BeforeEach
    void setUp() {
        activityRepository = mock(GroupBuyActivityRepository.class);
        teamRepository = mock(GroupBuyTeamRepository.class);
        orderRepository = mock(GroupBuyOrderRepository.class);
        orderPort = mock(OrderPort.class);
        eventPort = mock(GroupBuyEventPort.class);
        service = new GroupBuyApplicationService(
                activityRepository, teamRepository, orderRepository, orderPort, eventPort);
    }

    @Test
    void createActivitySucceedsWithValidParameters() {
        Long activityId = assertDoesNotThrow(() -> service.createActivity(
                "Test Group Buy",
                100L,
                "Test Product",
                new BigDecimal("100.00"),
                new BigDecimal("80.00"),
                3,
                24,
                LocalDateTime.now(),
                LocalDateTime.now().plusDays(7)));
        assertNotNull(activityId);
    }

    @Test
    void createGroupSucceedsAsNewGroup() {
        GroupBuyActivityDO activity = GroupBuyActivityDO.builder()
                .id(1L)
                .activityName("Test")
                .productId(100L)
                .productName("Product")
                .singlePrice(new BigDecimal("100.00"))
                .groupPrice(new BigDecimal("80.00"))
                .requiredQuantity(3)
                .currentQuantity(0)
                .status(1)
                .validityHours(24)
                .startTime(LocalDateTime.now().minusDays(1))
                .endTime(LocalDateTime.now().plusDays(7))
                .build();

        when(activityRepository.findById(1L)).thenReturn(activity);
        when(orderPort.createOrder(any(), any(), any(), any(), any())).thenReturn(1000L);
        when(activityRepository.findById(1L)).thenReturn(activity);

        Long teamId = service.createGroup(1L, 1L, "Test remark");

        assertNotNull(teamId);
        verify(teamRepository).save(any(GroupBuyTeamDO.class));
        verify(orderPort).createOrder(any(), any(), any(), any(), any());
    }

    @Test
    void createGroupFailsWhenActivityNotFound() {
        when(activityRepository.findById(999L)).thenReturn(null);

        assertThrows(IllegalArgumentException.class, () -> service.createGroup(1L, 999L, "remark"));
    }

    @Test
    void createGroupFailsWhenActivityInactive() {
        GroupBuyActivityDO activity = GroupBuyActivityDO.builder()
                .id(1L)
                .status(0)
                .build();
        when(activityRepository.findById(1L)).thenReturn(activity);

        assertThrows(IllegalStateException.class, () -> service.createGroup(1L, 1L, "remark"));
    }

    @Test
    void joinGroupSucceedsAsNewMember() {
        GroupBuyActivityDO activity = GroupBuyActivityDO.builder()
                .id(1L)
                .productId(100L)
                .productName("Product")
                .groupPrice(new BigDecimal("80.00"))
                .requiredQuantity(3)
                .status(1)
                .startTime(LocalDateTime.now().minusDays(1))
                .endTime(LocalDateTime.now().plusDays(7))
                .build();

        GroupBuyTeamDO team = GroupBuyTeamDO.builder()
                .id(1L)
                .activityId(1L)
                .leaderId(1L)
                .requiredQuantity(3)
                .currentQuantity(1)
                .status("PENDING")
                .expireTime(LocalDateTime.now().plusHours(24))
                .build();

        when(teamRepository.findById(1L)).thenReturn(team);
        when(activityRepository.findById(1L)).thenReturn(activity);
        when(orderRepository.findByTeamIdAndUserId(1L, 2L)).thenReturn(null);
        when(orderPort.createOrder(any(), any(), any(), any(), any())).thenReturn(1001L);

        Long orderId = service.joinGroup(2L, 1L, "Join group");

        assertNotNull(orderId);
        verify(orderRepository).save(any(GroupBuyOrderDO.class));
    }

    @Test
    void joinGroupFailsWhenTeamExpired() {
        GroupBuyTeamDO team = GroupBuyTeamDO.builder()
                .id(1L)
                .activityId(1L)
                .status("PENDING")
                .expireTime(LocalDateTime.now().minusHours(1))
                .build();

        when(teamRepository.findById(1L)).thenReturn(team);

        assertThrows(IllegalStateException.class, () -> service.joinGroup(2L, 1L, "remark"));
    }

    @Test
    void joinGroupFailsWhenUserAlreadyInTeam() {
        GroupBuyActivityDO activity = GroupBuyActivityDO.builder()
                .id(1L)
                .productId(100L)
                .groupPrice(new BigDecimal("80.00"))
                .requiredQuantity(3)
                .status(1)
                .startTime(LocalDateTime.now().minusDays(1))
                .endTime(LocalDateTime.now().plusDays(7))
                .build();

        GroupBuyTeamDO team = GroupBuyTeamDO.builder()
                .id(1L)
                .activityId(1L)
                .status("PENDING")
                .expireTime(LocalDateTime.now().plusHours(24))
                .build();

        GroupBuyOrderDO existingOrder = GroupBuyOrderDO.builder()
                .id(1L)
                .teamId(1L)
                .userId(1L)
                .build();

        when(teamRepository.findById(1L)).thenReturn(team);
        when(activityRepository.findById(1L)).thenReturn(activity);
        when(orderRepository.findByTeamIdAndUserId(1L, 1L)).thenReturn(existingOrder);

        assertThrows(IllegalStateException.class, () -> service.joinGroup(1L, 1L, "remark"));
    }

    @Test
    void getActiveActivitiesReturnsActiveList() {
        List<GroupBuyActivityDO> result = service.getActiveActivities();
        assertNotNull(result);
        verify(activityRepository).findActiveActivities();
    }

    private void when(java.lang.reflect.InvocationOnMock invocation) {
    }
}