package com.metawebthree.finance.application.command.ledger.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;


@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class LedgerGenerateCommand {
    private Long voucherId;
    private String voucherNo;
    private String createdBy;
}