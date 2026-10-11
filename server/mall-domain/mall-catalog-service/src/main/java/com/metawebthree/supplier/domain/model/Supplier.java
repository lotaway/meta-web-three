package com.metawebthree.supplier.domain.model;

import java.time.LocalDateTime;

public class Supplier {

    private Long id;
    private String supplierCode;
    private String supplierName;
    private String contactPerson;
    private String contactPhone;
    private String contactEmail;
    private String address;
    private SupplierStatus status;
    private VerificationStatus verificationStatus;
    private String businessLicense;
    private String legalPerson;
    private Integer supplierLevel;
    private Integer score;
    private String remark;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;

    public enum SupplierStatus {
        PENDING(0),
        ACTIVE(1),
        SUSPENDED(2),
        DISABLED(3);

        private final Integer value;

        SupplierStatus(Integer value) {
            this.value = value;
        }

        public Integer getValue() {
            return value;
        }

        public static SupplierStatus fromValue(Integer value) {
            for (SupplierStatus status : values()) {
                if (status.value.equals(value)) {
                    return status;
                }
            }
            return PENDING;
        }
    }

    public enum VerificationStatus {
        NOT_SUBMITTED(0),
        PENDING(1),
        APPROVED(2),
        REJECTED(3);

        private final Integer value;

        VerificationStatus(Integer value) {
            this.value = value;
        }

        public Integer getValue() {
            return value;
        }

        public static VerificationStatus fromValue(Integer value) {
            for (VerificationStatus status : values()) {
                if (status.value.equals(value)) {
                    return status;
                }
            }
            return NOT_SUBMITTED;
        }
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getSupplierCode() {
        return supplierCode;
    }

    public void setSupplierCode(String supplierCode) {
        this.supplierCode = supplierCode;
    }

    public String getSupplierName() {
        return supplierName;
    }

    public void setSupplierName(String supplierName) {
        this.supplierName = supplierName;
    }

    public String getContactPerson() {
        return contactPerson;
    }

    public void setContactPerson(String contactPerson) {
        this.contactPerson = contactPerson;
    }

    public String getContactPhone() {
        return contactPhone;
    }

    public void setContactPhone(String contactPhone) {
        this.contactPhone = contactPhone;
    }

    public String getContactEmail() {
        return contactEmail;
    }

    public void setContactEmail(String contactEmail) {
        this.contactEmail = contactEmail;
    }

    public String getAddress() {
        return address;
    }

    public void setAddress(String address) {
        this.address = address;
    }

    public SupplierStatus getStatus() {
        return status;
    }

    public void setStatus(SupplierStatus status) {
        this.status = status;
    }

    public VerificationStatus getVerificationStatus() {
        return verificationStatus;
    }

    public void setVerificationStatus(VerificationStatus verificationStatus) {
        this.verificationStatus = verificationStatus;
    }

    public String getBusinessLicense() {
        return businessLicense;
    }

    public void setBusinessLicense(String businessLicense) {
        this.businessLicense = businessLicense;
    }

    public String getLegalPerson() {
        return legalPerson;
    }

    public void setLegalPerson(String legalPerson) {
        this.legalPerson = legalPerson;
    }

    public Integer getSupplierLevel() {
        return supplierLevel;
    }

    public void setSupplierLevel(Integer supplierLevel) {
        this.supplierLevel = supplierLevel;
    }

    public Integer getScore() {
        return score;
    }

    public void setScore(Integer score) {
        this.score = score;
    }

    public String getRemark() {
        return remark;
    }

    public void setRemark(String remark) {
        this.remark = remark;
    }

    public LocalDateTime getCreateTime() {
        return createTime;
    }

    public void setCreateTime(LocalDateTime createTime) {
        this.createTime = createTime;
    }

    public LocalDateTime getUpdateTime() {
        return updateTime;
    }

    public void setUpdateTime(LocalDateTime updateTime) {
        this.updateTime = updateTime;
    }

    public void approve() {
        this.verificationStatus = VerificationStatus.APPROVED;
        this.status = SupplierStatus.ACTIVE;
    }

    public void reject(String reason) {
        this.verificationStatus = VerificationStatus.REJECTED;
        this.remark = reason;
    }

    public void submitForVerification() {
        if (this.verificationStatus == VerificationStatus.NOT_SUBMITTED) {
            this.verificationStatus = VerificationStatus.PENDING;
        }
    }

    public void updateScore(Integer delta) {
        this.score = (this.score == null ? 0 : this.score) + delta;
        if (this.score > 100) {
            this.score = 100;
        } else if (this.score < 0) {
            this.score = 0;
        }
    }
}