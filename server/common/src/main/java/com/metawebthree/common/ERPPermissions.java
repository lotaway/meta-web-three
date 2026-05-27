package com.metawebthree.common;

public final class ERPPermissions {

    private ERPPermissions() {}

    public static final String ACCOUNT_READ   = "erp:account:read";
    public static final String ACCOUNT_CREATE = "erp:account:create";
    public static final String ACCOUNT_UPDATE = "erp:account:update";
    public static final String ACCOUNT_DELETE = "erp:account:delete";

    public static final String ACCOUNT_SUBJECT_READ   = "erp:account-subject:read";
    public static final String ACCOUNT_SUBJECT_CREATE = "erp:account-subject:create";
    public static final String ACCOUNT_SUBJECT_UPDATE = "erp:account-subject:update";
    public static final String ACCOUNT_SUBJECT_DELETE = "erp:account-subject:delete";

    public static final String VOUCHER_READ   = "erp:voucher:read";
    public static final String VOUCHER_CREATE = "erp:voucher:create";
    public static final String VOUCHER_APPROVE = "erp:voucher:approve";
    public static final String VOUCHER_DELETE = "erp:voucher:delete";

    public static final String REPORT_READ  = "erp:report:read";
    public static final String REPORT_EXPORT = "erp:report:export";

    public static final String INVOICE_READ   = "erp:invoice:read";
    public static final String INVOICE_CREATE = "erp:invoice:create";
    public static final String INVOICE_ISSUE  = "erp:invoice:issue";
    public static final String INVOICE_PRINT  = "erp:invoice:print";
    public static final String INVOICE_VOID   = "erp:invoice:void";
    public static final String INVOICE_RED_FLUSH = "erp:invoice:red-flush";
    public static final String INVOICE_DELETE = "erp:invoice:delete";

    public static final String SETTLEMENT_READ    = "erp:settlement:read";
    public static final String SETTLEMENT_CREATE  = "erp:settlement:create";
    public static final String SETTLEMENT_CONFIRM = "erp:settlement:confirm";
    public static final String SETTLEMENT_PROCESS = "erp:settlement:process";
    public static final String SETTLEMENT_COMPLETE = "erp:settlement:complete";
    public static final String SETTLEMENT_FAIL    = "erp:settlement:fail";
    public static final String SETTLEMENT_CANCEL  = "erp:settlement:cancel";
    public static final String SETTLEMENT_REFUND  = "erp:settlement:refund";

    public static final String SALES_REPORT_READ   = "erp:sales-report:read";
    public static final String SALES_REPORT_CREATE = "erp:sales-report:create";
    public static final String SALES_REPORT_EXPORT = "erp:sales-report:export";

    public static final String INVENTORY_REPORT_READ   = "erp:inventory-report:read";
    public static final String INVENTORY_REPORT_CREATE = "erp:inventory-report:create";
    public static final String INVENTORY_REPORT_EXPORT = "erp:inventory-report:export";

    public static final String FINANCIAL_REPORT_READ   = "erp:financial-report:read";
    public static final String FINANCIAL_REPORT_CREATE = "erp:financial-report:create";
    public static final String FINANCIAL_REPORT_EXPORT = "erp:financial-report:export";
}
