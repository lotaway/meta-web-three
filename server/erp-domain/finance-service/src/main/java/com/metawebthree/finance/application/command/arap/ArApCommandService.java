package com.metawebthree.finance.application.command.arap;

import com.metawebthree.finance.application.command.arap.dto.ArCreateCommand;
import com.metawebthree.finance.application.command.arap.dto.ArReceiveCommand;
import com.metawebthree.finance.application.command.arap.dto.ApCreateCommand;
import com.metawebthree.finance.application.command.arap.dto.ApPayCommand;
import com.metawebthree.finance.domain.entity.arap.AccountsReceivable;
import com.metawebthree.finance.domain.entity.arap.AccountsPayable;
import com.metawebthree.finance.domain.exception.ArNotFoundException;
import com.metawebthree.finance.domain.exception.ApNotFoundException;
import com.metawebthree.finance.domain.repository.arap.AccountsReceivableRepository;
import com.metawebthree.finance.domain.repository.arap.AccountsPayableRepository;
import org.springframework.stereotype.Service;
import java.time.LocalDate;

@Service
public class ArApCommandService {
    private final AccountsReceivableRepository arRepository;
    private final AccountsPayableRepository apRepository;

    public ArApCommandService(AccountsReceivableRepository arRepository,
                              AccountsPayableRepository apRepository) {
        this.arRepository = arRepository;
        this.apRepository = apRepository;
    }

    public AccountsReceivable createAr(ArCreateCommand command) {
        AccountsReceivable ar = new AccountsReceivable();
        LocalDate invoiceDate = command.getInvoiceDate() != null ? command.getInvoiceDate() : LocalDate.now();
        Integer creditTerm = command.getCreditTerm() != null ? command.getCreditTerm() : 30;
        
        ar.create(command.getArCode(), command.getCustomerId(), command.getCustomerName(),
                  command.getBusinessType(), command.getAmount(), invoiceDate,
                  creditTerm, command.getCurrency(), command.getCreatedBy(), command.getCreatorName());
        
        if (command.getRelatedDocumentType() != null && command.getRelatedDocumentNo() != null) {
            ar.updateRelatedDocument(command.getRelatedDocumentType(), command.getRelatedDocumentNo());
        }
        ar.setDescription(command.getDescription());
        
        return arRepository.save(ar);
    }

    public AccountsReceivable receiveAr(ArReceiveCommand command) {
        AccountsReceivable ar = arRepository.findById(command.getArId())
            .orElseThrow(() -> new ArNotFoundException(command.getArId()));
        
        ar.receive(command.getReceiveAmount());
        
        return arRepository.save(ar);
    }

    public AccountsReceivable writeOffAr(Long arId, java.math.BigDecimal amount) {
        AccountsReceivable ar = arRepository.findById(arId)
            .orElseThrow(() -> new ArNotFoundException(arId));
        
        ar.writeOff(amount);
        
        return arRepository.save(ar);
    }

    public AccountsPayable createAp(ApCreateCommand command) {
        AccountsPayable ap = new AccountsPayable();
        LocalDate invoiceDate = command.getInvoiceDate() != null ? command.getInvoiceDate() : LocalDate.now();
        Integer creditTerm = command.getCreditTerm() != null ? command.getCreditTerm() : 30;
        
        ap.create(command.getApCode(), command.getSupplierId(), command.getSupplierName(),
                  command.getBusinessType(), command.getAmount(), invoiceDate,
                  creditTerm, command.getCurrency(), command.getCreatedBy(), command.getCreatorName());
        
        if (command.getRelatedDocumentType() != null && command.getRelatedDocumentNo() != null) {
            ap.updateRelatedDocument(command.getRelatedDocumentType(), command.getRelatedDocumentNo());
        }
        ap.setDescription(command.getDescription());
        
        return apRepository.save(ap);
    }

    public AccountsPayable payAp(ApPayCommand command) {
        AccountsPayable ap = apRepository.findById(command.getApId())
            .orElseThrow(() -> new ApNotFoundException(command.getApId()));
        
        ap.pay(command.getPayAmount());
        
        return apRepository.save(ap);
    }

    public AccountsPayable writeOffAp(Long apId, java.math.BigDecimal amount) {
        AccountsPayable ap = apRepository.findById(apId)
            .orElseThrow(() -> new ApNotFoundException(apId));
        
        ap.writeOff(amount);
        
        return apRepository.save(ap);
    }

    public void checkOverdueAr() {
        arRepository.findAll().forEach(ar -> {
            ar.checkOverdue();
            arRepository.save(ar);
        });
    }

    public void checkOverdueAp() {
        apRepository.findAll().forEach(ap -> {
            ap.checkOverdue();
            apRepository.save(ap);
        });
    }
}