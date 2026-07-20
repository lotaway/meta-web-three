package com.metawebthree.reporting.application;

import com.metawebthree.reporting.domain.entity.ReportSubscription;
import com.metawebthree.reporting.domain.entity.ReportSubscription.Channel;
import com.metawebthree.reporting.domain.entity.ReportSubscription.Frequency;
import com.metawebthree.reporting.domain.entity.ReportSubscription.ReportType;
import com.metawebthree.reporting.domain.repository.FinancialReportRepository;
import com.metawebthree.reporting.domain.repository.InventoryReportRepository;
import com.metawebthree.reporting.domain.repository.SalesReportRepository;
import com.metawebthree.reporting.domain.service.ReportSubscriptionService;
import jakarta.mail.internet.MimeMessage;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.web.client.RestTemplate;

import java.lang.reflect.Field;
import java.util.List;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class ReportDeliveryServiceTest {

    @Mock
    private ReportSubscriptionService subscriptionService;
    @Mock
    private SalesReportRepository salesReportRepository;
    @Mock
    private InventoryReportRepository inventoryReportRepository;
    @Mock
    private FinancialReportRepository financialReportRepository;
    @Mock
    private RestTemplate restTemplate;
    @Mock
    private JavaMailSender mailSender;
    @Mock
    private MimeMessage mimeMessage;

    private ReportDeliveryService service;

    @BeforeEach
    void setUp() throws Exception {
        service = new ReportDeliveryService(subscriptionService, salesReportRepository,
                inventoryReportRepository, financialReportRepository, restTemplate, mailSender);

        Field emailFromField = ReportDeliveryService.class.getDeclaredField("emailFrom");
        emailFromField.setAccessible(true);
        emailFromField.set(service, "noreply@metawebthree.com");
    }

    @Test
    void processDueSubscriptions_withNoSubscriptions_shouldNotSendAnything() {
        when(subscriptionService.getDueSubscriptions()).thenReturn(List.of());

        service.processDueSubscriptions();

        verify(mailSender, never()).send(any(MimeMessage.class));
        verify(restTemplate, never()).postForEntity(anyString(), any(), any());
    }

    @Test
    void processDueSubscriptions_withEmailChannel_shouldSendEmail() {
        ReportSubscription subscription = new ReportSubscription();
        subscription.setId(1L);
        subscription.setUserId(100L);
        subscription.setUserName("TestUser");
        subscription.setReportType(ReportType.SALES);
        subscription.setFrequency(Frequency.DAILY);
        subscription.setChannel(Channel.EMAIL);
        subscription.setRecipient("test@example.com");

        when(subscriptionService.getDueSubscriptions()).thenReturn(List.of(subscription));
        when(mailSender.createMimeMessage()).thenReturn(mimeMessage);

        service.processDueSubscriptions();

        verify(mailSender).send((MimeMessage) mimeMessage);
        verify(subscriptionService).markAsSent(1L);
    }

    @Test
    void processDueSubscriptions_withDingTalkChannel_shouldSendDingTalk() {
        ReportSubscription subscription = new ReportSubscription();
        subscription.setId(2L);
        subscription.setUserId(200L);
        subscription.setUserName("DingUser");
        subscription.setReportType(ReportType.SALES);
        subscription.setFrequency(Frequency.DAILY);
        subscription.setChannel(Channel.DINGTALK);
        subscription.setWebhookUrl("https://oapi.dingtalk.com/robot/send?access_token=test");

        when(subscriptionService.getDueSubscriptions()).thenReturn(List.of(subscription));
        when(restTemplate.postForEntity(anyString(), any(), eq(String.class)))
                .thenReturn(new ResponseEntity<>(HttpStatus.OK));

        service.processDueSubscriptions();

        verify(restTemplate).postForEntity(eq("https://oapi.dingtalk.com/robot/send?access_token=test"),
                any(), eq(String.class));
        verify(subscriptionService).markAsSent(2L);
    }
}
