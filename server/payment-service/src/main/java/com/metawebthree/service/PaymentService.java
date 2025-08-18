package com.metawebthree.service;

import com.metawebthree.entity.ExchangeOrder;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.UUID;

public interface PaymentService {
    
    public String createPayment(ExchangeOrder order);
    
    public boolean verifyPaymentCallback(String paymentOrderNo, String signature, String data);
    
    public String queryPaymentStatus(String paymentOrderNo);
    
    public boolean refundPayment(String paymentOrderNo, String reason);
} 