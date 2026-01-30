package com.metawebthree.payment.domain.model;

import com.baomidou.mybatisplus.annotation.*;
import com.metawebthree.common.DO.BaseDO;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;
import lombok.AllArgsConstructor;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
@SuperBuilder
@TableName("User_Kyc")
public class UserKYC extends BaseDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    @TableField("user_id")
    private Long userId;
    
    @TableField("level")
    private KYCLevel level;
    
    @TableField("status")
    private KYCStatus status;
    
    @TableField("real_name")
    private String realName;
    
    @TableField("id_number")
    private String idNumber;
    
    @TableField("id_type")
    private String idType; // ID_CARD, PASSPORT, DRIVER_LICENSE
    
    @TableField("phone_number")
    private String phoneNumber;
    
    @TableField("email")
    private String email;
    
    @TableField("address")
    private String address;
    
    @TableField("country")
    private String country;
    
    @TableField("nationality")
    private String nationality;
    
    @TableField("date_of_birth")
    private String dateOfBirth;
    
    @TableField("gender")
    private String gender;
    
    @TableField("id_card_front_url")
    private String idCardFrontUrl;
    
    @TableField("id_card_back_url")
    private String idCardBackUrl;
    
    @TableField("selfie_url")
    private String selfieUrl;
    
    @TableField("proof_of_address_url")
    private String proofOfAddressUrl;
    
    @TableField("bank_account_number")
    private String bankAccountNumber;
    
    @TableField("bank_name")
    private String bankName;
    
    @TableField("bank_branch")
    private String bankBranch;
    
    @TableField("tax_id")
    private String taxId;
    
    @TableField("occupation")
    private String occupation;
    
    @TableField("employer")
    private String employer;
    
    @TableField("annual_income")
    private String annualIncome;
    
    @TableField("source_of_funds")
    private String sourceOfFunds;
    
    @TableField("purpose_of_transaction")
    private String purposeOfTransaction;
    
    @TableField("reviewer_id")
    private String reviewerId;
    
    @TableField("review_notes")
    private String reviewNotes;
    
    @TableField("submitted_at")
    private LocalDateTime submittedAt;
    
    @TableField("reviewed_at")
    private LocalDateTime reviewedAt;
    
    public enum KYCLevel {
        L0("基础验证", 1000),
        L1("身份验证", 10000),
        L2("高级验证", 100000),
        L3("企业验证", 1000000);
        
        private final String description;
        private final int limit;
        
        KYCLevel(String description, int limit) {
            this.description = description;
            this.limit = limit;
        }
        
        public String getDescription() {
            return description;
        }
        
        public int getLimit() {
            return limit;
        }
    }
    
    public enum KYCStatus {
        PENDING("待审核"),
        APPROVED("已通过"),
        REJECTED("已拒绝"),
        EXPIRED("已过期");
        
        private final String description;
        
        KYCStatus(String description) {
            this.description = description;
        }
        
        public String getDescription() {
            return description;
        }
    }
} 