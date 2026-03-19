package com.metawebthree.common.services;

import java.util.ArrayList;
import java.util.List;

import org.quartz.CronScheduleBuilder;
import org.quartz.CronTrigger;
import org.quartz.Job;
import org.quartz.JobBuilder;
import org.quartz.JobDetail;
import org.quartz.JobKey;
import org.quartz.ScheduleBuilder;
import org.quartz.Scheduler;
import org.quartz.SchedulerException;
import org.quartz.Trigger;
import org.quartz.TriggerBuilder;
import org.quartz.TriggerKey;
import org.springframework.stereotype.Service;

import com.metawebthree.common.annotations.LogMethod;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Service
@RequiredArgsConstructor
public class QuartzManager {

    private final Scheduler scheduler;

    public Class<? extends Job> tryConvertClassIntoJob(Class<?> clazz) {
        if (!Job.class.isAssignableFrom(clazz)) {
            throw new IllegalArgumentException("The class " + clazz.getName() + " does not implement Job.");
        }
        @SuppressWarnings("unchecked")
        Class<? extends Job> jobClass = (Class<? extends Job>) clazz;
        return jobClass;
    }

    public Class<?> tryConvertClassNameIntoClass(String className) throws ClassNotFoundException {
        Class<?> clazz = null;
        if (className.indexOf(".") > -1) {
            clazz = Class.forName(className);
        } else {
            clazz = Class.forName("com.metawebthree.job." + className);
        }
        return clazz;
    }

    public QuartzManager addJob(String jobName, String jobGroupName, String triggerName, String triggerGroupName,
            String jobClassName, String cron, String description, String data) {
        try {
            Class<?> jobClass = tryConvertClassNameIntoClass(jobClassName);
            addJob(jobName, jobGroupName, triggerName, triggerGroupName, jobClass, cron, description, data);
        } catch (Exception e) {
            throw new JobScheduleException(jobName, jobGroupName, e);
        }
        return this;
    }

    public QuartzManager addJob(String jobName, String jobGroupName, String triggerName, String triggerGroupName,
            Class<?> jobClass, String cron, String description, String data) {
        return addJob(jobName, jobGroupName, triggerName, triggerGroupName, jobClass,
                CronScheduleBuilder.cronSchedule(cron), description, data);
    }

    public <SBT extends Trigger> QuartzManager addJob(String jobName, String jobGroupName, String triggerName,
            String triggerGroupName,
            Class<?> jobClass, ScheduleBuilder<SBT> schedBuilder, String description, String data) {
        try {
            JobDetail jobDetail = JobBuilder.newJob(tryConvertClassIntoJob(jobClass))
                    .withIdentity(jobName, jobGroupName)
                    .withDescription(description).storeDurably().requestRecovery().build();
            jobDetail.getJobDataMap().put("extraData", data);
            TriggerBuilder<Trigger> triggerBuilder = TriggerBuilder.newTrigger();
            triggerBuilder.withIdentity(triggerName, triggerGroupName);
            triggerBuilder.startNow();
            triggerBuilder.withDescription(description);
            triggerBuilder.withSchedule(schedBuilder);
            CronTrigger trigger = (CronTrigger) triggerBuilder.build();
            scheduler.scheduleJob(jobDetail, trigger);
            if (!scheduler.isShutdown()) {
                scheduler.start();
            }
        } catch (Exception e) {
            throw new JobScheduleException(jobName, jobGroupName, e);
        }
        return this;
    }

    public void modifyJobTime(String triggerName, String triggerGroupName,
            String cron) {
        try {
            TriggerKey triggerKey = TriggerKey.triggerKey(triggerName, triggerGroupName);
            CronTrigger trigger = (CronTrigger) scheduler.getTrigger(triggerKey);
            if (trigger == null) {
                return;
            }
            String oldTime = trigger.getCronExpression();
            if (!oldTime.equalsIgnoreCase(cron)) {
                TriggerBuilder<Trigger> triggerBuilder = TriggerBuilder.newTrigger();
                triggerBuilder.withIdentity(triggerName, triggerGroupName);
                triggerBuilder.startNow();
                triggerBuilder.withDescription(trigger.getDescription());
                triggerBuilder.withSchedule(CronScheduleBuilder.cronSchedule(cron));
                trigger = (CronTrigger) triggerBuilder.build();
                scheduler.rescheduleJob(triggerKey, trigger);
            }
        } catch (Exception e) {
            throw new JobScheduleException(triggerName, triggerGroupName, e);
        }
    }

    @LogMethod
    public void removeJob(String jobName, String jobGroupName, String triggerName, String triggerGroupName) {
        try {
            TriggerKey triggerKey = TriggerKey.triggerKey(triggerName, triggerGroupName);
            scheduler.pauseTrigger(triggerKey);
            scheduler.unscheduleJob(triggerKey);
            scheduler.deleteJob(JobKey.jobKey(jobName, jobGroupName));
        } catch (SchedulerException e) {
            log.error("Failed to remove job - jobName: {}, group: {}", jobName, jobGroupName, e);
            throw new JobRemovalException(jobName, jobGroupName, e);
        }
    }

    public void pauseJob(String triggerName, String triggerGroupName) {
        try {
            TriggerKey triggerKey = TriggerKey.triggerKey(triggerName, triggerGroupName);
            scheduler.pauseTrigger(triggerKey);
        } catch (SchedulerException e) {
            log.error("Failed to pause job - trigger: {}", triggerName, e);
            throw new JobOperationException("pause", triggerName, e);
        }
    }

    public void startJob(String triggerName, String triggerGroupName) {
        try {
            TriggerKey triggerKey = TriggerKey.triggerKey(triggerName, triggerGroupName);
            scheduler.resumeTrigger(triggerKey);
        } catch (SchedulerException e) {
            log.error("Failed to start job - trigger: {}", triggerName, e);
            throw new JobOperationException("start", triggerName, e);
        }
    }

    public void startAllJobs() {
        try {
            scheduler.start();
        } catch (SchedulerException e) {
            throw new JobScheduleException("all", "jobs", e);
        }
    }

    public void shutdownAllJobs() {
        try {
            if (!scheduler.isShutdown()) {
                scheduler.shutdown();
            }
        } catch (SchedulerException e) {
            throw new JobScheduleException("all", "shutdown", e);
        }
    }
}

class JobScheduleException extends RuntimeException {
    private final String jobName;
    private final String jobGroup;

    public JobScheduleException(String jobName, String jobGroup, Throwable cause) {
        super("Failed to schedule job: " + jobName, cause);
        this.jobName = jobName;
        this.jobGroup = jobGroup;
    }
}

class JobRemovalException extends RuntimeException {
    private final String jobName;
    private final String jobGroup;

    public JobRemovalException(String jobName, String jobGroup, Throwable cause) {
        super("Failed to remove job: " + jobName, cause);
        this.jobName = jobName;
        this.jobGroup = jobGroup;
    }
}

class JobOperationException extends RuntimeException {
    private final String operation;
    private final String triggerName;

    public JobOperationException(String operation, String triggerName, Throwable cause) {
        super("Failed to " + operation + " job: " + triggerName, cause);
        this.operation = operation;
        this.triggerName = triggerName;
    }
}