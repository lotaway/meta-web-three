package com.metawebthree.crm.application.query;

import com.metawebthree.crm.domain.entity.Campaign;
import com.metawebthree.crm.domain.repository.CampaignRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class CampaignQueryServiceTest {

    @Mock private CampaignRepository campaignRepository;

    private CampaignQueryService service;

    @BeforeEach
    void setUp() {
        service = new CampaignQueryService(campaignRepository);
    }

    @Test
    void getById_whenFound_shouldReturnCampaign() {
        Campaign campaign = new Campaign();
        campaign.setId(1L);
        campaign.setName("Test Campaign");
        when(campaignRepository.findById(1L)).thenReturn(Optional.of(campaign));

        Campaign result = service.getById(1L);

        assertSame(campaign, result);
        verify(campaignRepository).findById(1L);
    }

    @Test
    void getById_whenNotFound_shouldReturnNull() {
        when(campaignRepository.findById(99L)).thenReturn(Optional.empty());

        Campaign result = service.getById(99L);

        assertNull(result);
        verify(campaignRepository).findById(99L);
    }

    @Test
    void listAll_shouldReturnAll() {
        List<Campaign> campaigns = List.of(new Campaign(), new Campaign());
        when(campaignRepository.findAll()).thenReturn(campaigns);

        List<Campaign> result = service.listAll();

        assertSame(campaigns, result);
        verify(campaignRepository).findAll();
    }

    @Test
    void listByStatus_shouldDelegate() {
        List<Campaign> campaigns = List.of(new Campaign());
        when(campaignRepository.findByStatus("ACTIVE")).thenReturn(campaigns);

        List<Campaign> result = service.listByStatus("ACTIVE");

        assertSame(campaigns, result);
        verify(campaignRepository).findByStatus("ACTIVE");
    }

    @Test
    void listByType_shouldDelegate() {
        List<Campaign> campaigns = List.of(new Campaign());
        when(campaignRepository.findByType("EMAIL")).thenReturn(campaigns);

        List<Campaign> result = service.listByType("EMAIL");

        assertSame(campaigns, result);
        verify(campaignRepository).findByType("EMAIL");
    }
}
