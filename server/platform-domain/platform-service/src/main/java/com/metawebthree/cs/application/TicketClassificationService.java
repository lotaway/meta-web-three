package com.metawebthree.cs.application;

import com.metawebthree.cs.domain.model.enums.WorkOrderCategory;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

@Service
public class TicketClassificationService {
    
    private final Map<WorkOrderCategory, List<Pattern>> categoryPatterns = new HashMap<>();
    private Double lastConfidence = 0.0;
    
    public TicketClassificationService() {
        initializePatterns();
    }
    
    private void initializePatterns() {
        categoryPatterns.put(WorkOrderCategory.PRODUCT_INQUIRY, List.of(
            Pattern.compile(".*(价格|多少钱|规格|参数|尺寸|颜色|材质).*", Pattern.CASE_INSENSITIVE),
            Pattern.compile(".*(有没有货|何时上架|什么时候有).*", Pattern.CASE_INSENSITIVE),
            Pattern.compile(".*(怎么卖|优惠|折扣|活动).*", Pattern.CASE_INSENSITIVE)
        ));
        
        categoryPatterns.put(WorkOrderCategory.ORDER_INQUIRY, List.of(
            Pattern.compile(".*(订单|订单号|查询订单|订单状态).*", Pattern.CASE_INSENSITIVE),
            Pattern.compile(".*(什么时候发货|发货了吗|物流).*", Pattern.CASE_INSENSITIVE),
            Pattern.compile(".*(取消订单|修改订单).*", Pattern.CASE_INSENSITIVE)
        ));
        
        categoryPatterns.put(WorkOrderCategory.PAYMENT_ISSUE, List.of(
            Pattern.compile(".*(支付|付款|退款|钱款).*", Pattern.CASE_INSENSITIVE),
            Pattern.compile(".*(支付失败|无法支付|支付异常).*", Pattern.CASE_INSENSITIVE),
            Pattern.compile(".*(发票|开发票|税).*", Pattern.CASE_INSENSITIVE)
        ));
        
        categoryPatterns.put(WorkOrderCategory.SHIPPING_ISSUE, List.of(
            Pattern.compile(".*(快递|物流|运输|送货).*", Pattern.CASE_INSENSITIVE),
            Pattern.compile(".*(还没到|延迟|超时|丢失).*", Pattern.CASE_INSENSITIVE),
            Pattern.compile(".*(地址|收货|配送).*", Pattern.CASE_INSENSITIVE)
        ));
        
        categoryPatterns.put(WorkOrderCategory.REFUND_REQUEST, List.of(
            Pattern.compile(".*(退货|退款|换货|售后).*", Pattern.CASE_INSENSITIVE),
            Pattern.compile(".*(不要了|不想要了|取消).*", Pattern.CASE_INSENSITIVE),
            Pattern.compile(".*(质量问题|损坏|不满意).*", Pattern.CASE_INSENSITIVE)
        ));
        
        categoryPatterns.put(WorkOrderCategory.COMPLAINT, List.of(
            Pattern.compile(".*(投诉|差评|举报|曝光).*", Pattern.CASE_INSENSITIVE),
            Pattern.compile(".*(态度|服务|恶劣|不满).*", Pattern.CASE_INSENSITIVE),
            Pattern.compile(".*(欺骗|虚假|欺诈|骗).*", Pattern.CASE_INSENSITIVE)
        ));
        
        categoryPatterns.put(WorkOrderCategory.TECHNICAL_SUPPORT, List.of(
            Pattern.compile(".*(无法登录|账号|密码|注册).*", Pattern.CASE_INSENSITIVE),
            Pattern.compile(".*(系统|页面|APP|小程序).*", Pattern.CASE_INSENSITIVE),
            Pattern.compile(".*(报错|错误|异常|闪退).*", Pattern.CASE_INSENSITIVE)
        ));
        
        categoryPatterns.put(WorkOrderCategory.ACCOUNT_ISSUE, List.of(
            Pattern.compile(".*(会员|积分|等级|权益).*", Pattern.CASE_INSENSITIVE),
            Pattern.compile(".*(优惠券|礼品卡|红包).*", Pattern.CASE_INSENSITIVE),
            Pattern.compile(".*(账户|账号|实名).*", Pattern.CASE_INSENSITIVE)
        ));
    }
    
    public WorkOrderCategory classify(String title, String description) {
        String fullText = (title + " " + (description != null ? description : "")).toLowerCase();
        
        Map<WorkOrderCategory, Integer> scores = new HashMap<>();
        
        for (Map.Entry<WorkOrderCategory, List<Pattern>> entry : categoryPatterns.entrySet()) {
            int score = 0;
            for (Pattern pattern : entry.getValue()) {
                if (pattern.matcher(fullText).matches()) {
                    score++;
                }
            }
            if (score > 0) {
                scores.put(entry.getKey(), score);
            }
        }
        
        if (scores.isEmpty()) {
            lastConfidence = 0.3;
            return WorkOrderCategory.OTHER;
        }
        
        WorkOrderCategory bestCategory = scores.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse(WorkOrderCategory.OTHER);
        
        int maxScore = scores.get(bestCategory);
        lastConfidence = Math.min(0.5 + (maxScore * 0.1), 0.95);
        
        return bestCategory;
    }
    
    public Double getConfidence() {
        return lastConfidence;
    }
    
    public Map<WorkOrderCategory, Double> classifyWithScores(String title, String description) {
        String fullText = (title + " " + (description != null ? description : "")).toLowerCase();
        
        Map<WorkOrderCategory, Double> scores = new HashMap<>();
        
        for (Map.Entry<WorkOrderCategory, List<Pattern>> entry : categoryPatterns.entrySet()) {
            double score = 0;
            for (Pattern pattern : entry.getValue()) {
                if (pattern.matcher(fullText).matches()) {
                    score += 1.0;
                }
            }
            if (score > 0) {
                scores.put(entry.getKey(), score);
            }
        }
        
        return scores;
    }
}