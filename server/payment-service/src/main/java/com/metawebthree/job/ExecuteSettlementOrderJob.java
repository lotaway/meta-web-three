package com.metawebthree.job;

import com.metawebthree.service.impl.SettlementServiceImpl;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.quartz.Job;
import org.quartz.JobExecutionContext;
import org.quartz.JobExecutionException;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
@Slf4j
public class ExecuteSettlementOrderJob implements Job {

    private final SettlementServiceImpl settlementService;

    @Override
    public void execute(JobExecutionContext context) throws JobExecutionException {
        try {
            settlementService.executeSettlement("Scheduled");
        } catch (Exception e) {
            log.error("Error executing settlement job", e);
            throw new JobExecutionException(e);
        }
    }
}
