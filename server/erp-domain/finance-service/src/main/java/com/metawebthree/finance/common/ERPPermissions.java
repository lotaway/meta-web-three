package com.metawebthree.finance.common;

public final class ERPPermissions {

    private ERPPermissions() {}

    // Account permissions
    public static final String ACCOUNT_READ   = "erp:account:read";
    public static final String ACCOUNT_CREATE = "erp:account:create";
    public static final String ACCOUNT_UPDATE = "erp:account:update";
    public static final String ACCOUNT_DELETE = "erp:account:delete";

    // Account Subject permissions
    public static final String ACCOUNT_SUBJECT_READ   = "erp:account-subject:read";
    public static final String ACCOUNT_SUBJECT_CREATE = "erp:account-subject:create";
    public static final String ACCOUNT_SUBJECT_UPDATE = "erp:account-subject:update";
    public static final String ACCOUNT_SUBJECT_DELETE = "erp:account-subject:delete";

    // Voucher permissions
    public static final String VOUCHER_READ   = "erp:voucher:read";
    public static final String VOUCHER_CREATE = "erp:voucher:create";
    public static final String VOUCHER_APPROVE = "erp:voucher:approve";
    public static final String VOUCHER_DELETE = "erp:voucher:delete";

    // Financial Report permissions
    public static final String REPORT_READ  = "erp:report:read";
    public static final String REPORT_EXPORT = "erp:report:export";
}