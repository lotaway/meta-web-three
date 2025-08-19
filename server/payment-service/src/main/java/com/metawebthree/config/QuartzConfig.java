package com.metawebthree.config;

import org.quartz.*;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import com.metawebthree.job.ExecuteSettlementOrderJob;
import com.metawebthree.job.SettlementDailyClearingJob;

@Configuration
public class QuartzConfig {

    @Bean
    public JobDetail dailyClearingJobDetail() {
        return JobBuilder.newJob()
                .ofType(SettlementDailyClearingJob.class)
                .withIdentity("settlementDailyClearingJob", "settlementGroup")
                .storeDurably()
                .build();
    }

    @Bean
    public Trigger dailyClearingJobTrigger() {
        // Daily at 2:00 AM
        CronScheduleBuilder scheduleBuilder = CronScheduleBuilder.cronSchedule("0 0 2 * * ?")
                .withMisfireHandlingInstructionFireAndProceed();

        return TriggerBuilder.newTrigger()
                .forJob(dailyClearingJobDetail())
                .withIdentity("dailyClearingTrigger", "settlementGroup")
                .withSchedule(scheduleBuilder)
                .build();
    }

    @Bean
    public JobDetail executeSettlementJobDetail() {
        return JobBuilder.newJob()
                .ofType(ExecuteSettlementOrderJob.class)
                .withIdentity("executeSettlementOrderJob", "settlementGroup")
                .storeDurably()
                .build();
    }

    @Bean
    public Trigger executeSettlementJobTrigger() {
        // Daily at 9:00 AM
        CronScheduleBuilder scheduleBuilder = CronScheduleBuilder.cronSchedule("0 0 9 * * ?")
                .withMisfireHandlingInstructionFireAndProceed();

        return TriggerBuilder.newTrigger()
                .forJob(executeSettlementJobDetail())
                .withIdentity("executeSettlementTrigger", "settlementGroup")
                .withSchedule(scheduleBuilder)
                .build();
    }
}
