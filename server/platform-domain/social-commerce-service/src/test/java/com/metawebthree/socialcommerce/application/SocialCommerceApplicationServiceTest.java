package com.metawebthree.socialcommerce.application;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.metawebthree.socialcommerce.domain.model.ShareRecordDO;
import com.metawebthree.socialcommerce.domain.model.DistributionRelationDO;
import com.metawebthree.socialcommerce.domain.model.CommunityDO;
import com.metawebthree.socialcommerce.domain.repository.ShareRewardConfigRepository;
import com.metawebthree.socialcommerce.domain.repository.ShareRecordRepository;
import com.metawebthree.socialcommerce.domain.repository.DistributionRelationRepository;
import com.metawebthree.socialcommerce.domain.repository.DistributionRewardRepository;
import com.metawebthree.socialcommerce.domain.repository.CommunityRepository;
import com.metawebthree.socialcommerce.domain.repository.CommunityMemberRepository;
import com.metawebthree.socialcommerce.domain.ports.CommissionPort;
import com.metawebthree.socialcommerce.domain.ports.UserPort;
import com.metawebthree.socialcommerce.domain.ports.SocialCommerceEventPort;

class SocialCommerceApplicationServiceTest {

    private ShareRewardConfigRepository configRepository;
    private ShareRecordRepository shareRecordRepository;
    private DistributionRelationRepository relationRepository;
    private DistributionRewardRepository rewardRepository;
    private CommunityRepository communityRepository;
    private CommunityMemberRepository memberRepository;
    private CommissionPort commissionPort;
    private UserPort userPort;
    private SocialCommerceEventPort eventPort;
    private SocialCommerceApplicationService service;

    @BeforeEach
    void setUp() {
        configRepository = mock(ShareRewardConfigRepository.class);
        shareRecordRepository = mock(ShareRecordRepository.class);
        relationRepository = mock(DistributionRelationRepository.class);
        rewardRepository = mock(DistributionRewardRepository.class);
        communityRepository = mock(CommunityRepository.class);
        memberRepository = mock(CommunityMemberRepository.class);
        commissionPort = mock(CommissionPort.class);
        userPort = mock(UserPort.class);
        eventPort = mock(SocialCommerceEventPort.class);
        service = new SocialCommerceApplicationService(
                configRepository,
                shareRecordRepository,
                relationRepository,
                rewardRepository,
                communityRepository,
                memberRepository,
                commissionPort,
                userPort,
                eventPort);
    }

    @Test
    void createShareRewardConfigSucceeds() {
        Long configId = assertDoesNotThrow(() -> service.createShareRewardConfig(
                "Test Config",
                1,
                new BigDecimal("10.00"),
                new BigDecimal("0.05"),
                100,
                new BigDecimal("1000.00"),
                LocalDateTime.now(),
                LocalDateTime.now().plusDays(30)));
        assertNotNull(configId);
    }

    @Test
    void createShareRecordSucceedsWithValidUser() {
        when(userPort.isValidUser(1L)).thenReturn(true);

        String shareUrl = service.createShareRecord(1L, 100L, "PRODUCT", "WECHAT");

        assertNotNull(shareUrl);
        assert shareUrl.startsWith("/share/");
        verify(shareRecordRepository).save(any(ShareRecordDO.class));
    }

    @Test
    void createShareRecordFailsWithInvalidUser() {
        when(userPort.isValidUser(999L)).thenReturn(false);

        assertThrows(IllegalArgumentException.class, 
                () -> service.createShareRecord(999L, 100L, "PRODUCT", "WECHAT"));
    }

    @Test
    void bindDistributionRelationSucceedsWithValidReferrer() {
        when(userPort.isValidUser(2L)).thenReturn(true);
        when(relationRepository.existsByUserId(1L)).thenReturn(false);
        when(relationRepository.findByUserId(2L)).thenReturn(null);

        Boolean result = service.bindDistributionRelation(1L, 2L);

        assertEquals(true, result);
        verify(relationRepository).save(any(DistributionRelationDO.class));
    }

    @Test
    void bindDistributionRelationFailsWhenBindToSelf() {
        assertThrows(IllegalArgumentException.class, 
                () -> service.bindDistributionRelation(1L, 1L));
    }

    @Test
    void bindDistributionRelationFailsWhenUserAlreadyHasReferrer() {
        when(relationRepository.existsByUserId(1L)).thenReturn(true);

        assertThrows(IllegalStateException.class, 
                () -> service.bindDistributionRelation(1L, 2L));
    }

    @Test
    void bindDistributionRelationFailsWhenExceedsMaxLevel() {
        when(userPort.isValidUser(2L)).thenReturn(true);
        when(relationRepository.existsByUserId(1L)).thenReturn(false);
        
        DistributionRelationDO level3Referrer = DistributionRelationDO.builder()
                .userId(2L)
                .level(3)
                .rootReferrerId(5L)
                .build();
        when(relationRepository.findByUserId(2L)).thenReturn(level3Referrer);

        assertThrows(IllegalStateException.class, 
                () -> service.bindDistributionRelation(1L, 2L));
    }

    @Test
    void createCommunitySucceedsWithValidOwner() {
        when(userPort.isValidUser(1L)).thenReturn(true);
        when(userPort.getUserNickname(1L)).thenReturn("Owner");

        Long communityId = service.createCommunity(
                "Test Community",
                "Test Description",
                1L,
                "http://avatar.url",
                100);

        assertNotNull(communityId);
        verify(communityRepository).save(any(CommunityDO.class));
        verify(memberRepository).save(any(com.metawebthree.socialcommerce.domain.model.CommunityMemberDO.class));
    }

    @Test
    void createCommunityFailsWithInvalidOwner() {
        when(userPort.isValidUser(999L)).thenReturn(false);

        assertThrows(IllegalArgumentException.class, 
                () -> service.createCommunity("Test", "Desc", 999L, null, 100));
    }

    @Test
    void joinCommunitySucceedsWithValidCommunity() {
        CommunityDO community = CommunityDO.builder()
                .id(1L)
                .memberCount(1)
                .maxMembers(100)
                .status("ACTIVE")
                .build();

        when(communityRepository.findById(1L)).thenReturn(community);
        when(memberRepository.existsByCommunityIdAndUserId(1L, 2L)).thenReturn(false);
        when(userPort.getUserNickname(2L)).thenReturn("Member");

        Boolean result = service.joinCommunity(1L, 2L);

        assertEquals(true, result);
        verify(memberRepository).save(any(com.metawebthree.socialcommerce.domain.model.CommunityMemberDO.class));
    }

    @Test
    void joinCommunityFailsWhenCommunityFull() {
        CommunityDO community = CommunityDO.builder()
                .id(1L)
                .memberCount(100)
                .maxMembers(100)
                .status("ACTIVE")
                .build();

        when(communityRepository.findById(1L)).thenReturn(community);

        assertThrows(IllegalStateException.class, 
                () -> service.joinCommunity(1L, 2L));
    }

    @Test
    void joinCommunityFailsWhenAlreadyMember() {
        CommunityDO community = CommunityDO.builder()
                .id(1L)
                .memberCount(1)
                .maxMembers(100)
                .status("ACTIVE")
                .build();

        when(communityRepository.findById(1L)).thenReturn(community);
        when(memberRepository.existsByCommunityIdAndUserId(1L, 1L)).thenReturn(true);

        assertThrows(IllegalStateException.class, 
                () -> service.joinCommunity(1L, 1L));
    }

    @Test
    void joinCommunityByCodeSucceedsWithValidCode() {
        CommunityDO community = CommunityDO.builder()
                .id(1L)
                .memberCount(1)
                .maxMembers(100)
                .status("ACTIVE")
                .build();

        when(communityRepository.findByInviteCode("VALID123")).thenReturn(community);
        when(memberRepository.existsByCommunityIdAndUserId(1L, 2L)).thenReturn(false);
        when(userPort.getUserNickname(2L)).thenReturn("Member");

        Boolean result = service.joinCommunityByCode("VALID123", 2L);

        assertEquals(true, result);
    }

    @Test
    void joinCommunityByCodeFailsWithInvalidCode() {
        when(communityRepository.findByInviteCode("INVALID")).thenReturn(null);

        assertThrows(IllegalArgumentException.class, 
                () -> service.joinCommunityByCode("INVALID", 2L));
    }

    @Test
    void leaveCommunitySucceedsAsMember() {
        com.metawebthree.socialcommerce.domain.model.CommunityMemberDO member = 
                com.metawebthree.socialcommerce.domain.model.CommunityMemberDO.builder()
                .id(1L)
                .communityId(1L)
                .userId(2L)
                .role("MEMBER")
                .build();

        when(memberRepository.findByCommunityIdAndUserId(1L, 2L)).thenReturn(member);

        Boolean result = service.leaveCommunity(1L, 2L);

        assertEquals(true, result);
        verify(memberRepository).delete(1L);
    }

    @Test
    void leaveCommunityFailsAsOwner() {
        com.metawebthree.socialcommerce.domain.model.CommunityMemberDO member = 
                com.metawebthree.socialcommerce.domain.model.CommunityMemberDO.builder()
                .id(1L)
                .communityId(1L)
                .userId(1L)
                .role("OWNER")
                .build();

        when(memberRepository.findByCommunityIdAndUserId(1L, 1L)).thenReturn(member);

        assertThrows(IllegalStateException.class, 
                () -> service.leaveCommunity(1L, 1L));
    }

    @Test
    void getUserShareRecordsReturnsRecords() {
        List<ShareRecordDO> result = service.getUserShareRecords(1L);
        assertNotNull(result);
        verify(shareRecordRepository).findBySharerId(1L);
    }

    @Test
    void getUserDistributionRelationsReturnsRelations() {
        List<DistributionRelationDO> result = service.getUserDistributionRelations(1L);
        assertNotNull(result);
        verify(relationRepository).findByReferrerId(1L);
    }

    @Test
    void getUserCommunitiesReturnsCommunities() {
        List<com.metawebthree.socialcommerce.domain.model.CommunityMemberDO> members = List.of();
        when(memberRepository.findByUserId(1L)).thenReturn(members);

        List<CommunityDO> result = service.getUserCommunities(1L);
        assertNotNull(result);
    }
}