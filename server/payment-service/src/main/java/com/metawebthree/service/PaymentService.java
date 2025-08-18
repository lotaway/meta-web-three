package com.metawebthree.service;

import com.metawebthree.entity.ExchangeOrder;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.UUID;

public interface PaymentService {
    
    public String createPayment(ExchangeOrder order);
    
    private String createAlipayPayment(ExchangeOrder order, String paymentOrderNo);
    
    private String createWechatPayment(ExchangeOrder order, String paymentOrderNo);
    
    private String createBankTransferPayment(ExchangeOrder order, String paymentOrderNo);
    
    private String createApplePayPayment(ExchangeOrder order, String paymentOrderNo);
    
    private String createGooglePayPayment(ExchangeOrder order, String paymentOrderNo);
    
    public boolean verifyPaymentCallback(String paymentOrderNo, String signature, String data);
    
    public String queryPaymentStatus(String paymentOrderNo);
    
    private String generatePaymentOrderNo();
    
    public boolean refundPayment(String paymentOrderNo, String reason);
} 