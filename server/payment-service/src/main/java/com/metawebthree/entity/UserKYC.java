package com.metawebthree.entity;

import jakarta.persistence.*;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.Builder;

import java.time.LocalDateTime;

@Entity
@Table(name = "user_kyc")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class UserKYC {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(nullable = false)
    private Long userId;
    
    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private KYCLevel level;
    
    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private KYCStatus status;
    
    @Column
    private String realName;
    
    @Column
    private String idNumber;
    
    @Column
    private String idType; // ID_CARD, PASSPORT, DRIVER_LICENSE
    
    @Column
    private String phoneNumber;
    
    @Column
    private String email;
    
    @Column
    private String address;
    
    @Column
    private String country;
    
    @Column
    private String nationality;
    
    @Column
    private String dateOfBirth;
    
    @Column
    private String gender;
    
    @Column
    private String idCardFrontUrl;
    
    @Column
    private String idCardBackUrl;
    
    @Column
    private String selfieUrl;
    
    @Column
    private String proofOfAddressUrl;
    
    @Column
    private String bankAccountNumber;
    
    @Column
    private String bankName;
    
    @Column
    private String bankBranch;
    
    @Column
    private String taxId;
    
    @Column
    private String occupation;
    
    @Column
    private String employer;
    
    @Column
    private String annualIncome;
    
    @Column
    private String sourceOfFunds;
    
    @Column
    private String purposeOfTransaction;
    
    @Column
    private String reviewerId;
    
    @Column
    private String reviewNotes;
    
    @Column
    private LocalDateTime submittedAt;
    
    @Column
    private LocalDateTime reviewedAt;
    
    @Column(nullable = false)
    private LocalDateTime createdAt;
    
    @Column
    private LocalDateTime updatedAt;
    
    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        if (status == null) {
            status = KYCStatus.PENDING;
        }
    }
    
    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }
    
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