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
    private String idType;
    
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
        L0(KYCLevelConstant.BASIC_VERIFICATION, KYCLevelConstant.L0_LIMIT),
        L1(KYCLevelConstant.IDENTITY_VERIFICATION, KYCLevelConstant.L1_LIMIT),
        L2(KYCLevelConstant.ADVANCED_VERIFICATION, KYCLevelConstant.L2_LIMIT),
        L3(KYCLevelConstant.ENTERPRISE_VERIFICATION, KYCLevelConstant.L3_LIMIT);
        
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
        PENDING(KYCStatusConstant.PENDING),
        APPROVED(KYCStatusConstant.APPROVED),
        REJECTED(KYCStatusConstant.REJECTED),
        EXPIRED(KYCStatusConstant.EXPIRED);
        
        private final String description;
        
        KYCStatus(String description) {
            this.description = description;
        }
        
        public String getDescription() {
            return description;
        }
    }
}

class KYCLevelConstant {
    static final String BASIC_VERIFICATION = "基础验证";
    static final String IDENTITY_VERIFICATION = "身份验证";
    static final String ADVANCED_VERIFICATION = "高级验证";
    static final String ENTERPRISE_VERIFICATION = "企业验证";
    static final int L0_LIMIT = 1000;
    static final int L1_LIMIT = 10000;
    static final int L2_LIMIT = 100000;
    static final int L3_LIMIT = 1000000;
}

class KYCStatusConstant {
    static final String PENDING = "待审核";
    static final String APPROVED = "已通过";
    static final String REJECTED = "已拒绝";
    static final String EXPIRED = "已过期";
} 