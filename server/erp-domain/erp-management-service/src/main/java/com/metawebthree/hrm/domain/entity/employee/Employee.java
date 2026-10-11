package com.metawebthree.hrm.domain.entity.employee;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
@TableName("hrm_employee")
public class Employee {

    @TableId(type = IdType.ASSIGN_ID)
    private Long id;

    private String employeeNo;

    private String name;

    private Integer gender;

    private String mobile;

    private String email;

    private String idCard;

    private LocalDate birthday;

    private String nativePlace;

    private String nation;

    private Integer maritalStatus;

    private String politicalStatus;

    private Integer education;

    private String graduateSchool;

    private String major;

    private LocalDate hireDate;

    private LocalDate formalDate;

    private Long departmentId;

    private Long positionId;

    private String workLocation;

    private Integer status;

    private LocalDate contractStartDate;

    private LocalDate contractEndDate;

    private String emergencyContact;

    private String emergencyPhone;

    private String bankAccount;

    private String bankName;

    private String socialSecurityNo;

    private String housingFundNo;

    private String photoUrl;

    private String remark;

    private Long tenantId;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createdAt;

    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updatedAt;

    @TableLogic
    private Integer deleted;
}