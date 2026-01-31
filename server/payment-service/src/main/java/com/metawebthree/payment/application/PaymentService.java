package com.metawebthree.payment.application;
import com.metawebthree.payment.domain.model.ExchangeOrder;
 
public interface PaymentService {
    
    public String createPayment(ExchangeOrder order);
    
    public boolean verifyPaymentCallback(String paymentOrderNo, String signature, String data);
    
    public String queryPaymentStatus(String paymentOrderNo);
    
    public boolean refundPayment(String paymentOrderNo, String reason);
} 
