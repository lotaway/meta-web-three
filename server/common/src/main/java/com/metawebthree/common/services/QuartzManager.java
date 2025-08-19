package com.metawebthree.common.services;

import java.util.ArrayList;
import java.util.List;

import org.quartz.CronScheduleBuilder;
import org.quartz.CronTrigger;
import org.quartz.Job;
import org.quartz.JobBuilder;
import org.quartz.JobDetail;
import org.quartz.JobKey;
import org.quartz.Scheduler;
import org.quartz.SchedulerException;
import org.quartz.Trigger;
import org.quartz.TriggerBuilder;
import org.quartz.TriggerKey;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.metawebthree.common.annotations.LogMethod;

import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class QuartzManager {

    private final Scheduler scheduler;

    public void addJob(String jobName, String jobGroupName, String triggerName, String triggerGroupName,
            Class<?> jobClass, String cron, String description, String data) {
        try {
            if (!Job.class.isAssignableFrom(jobClass)) {
                throw new IllegalArgumentException("The class " + jobClass.getName() + " does not implement Job.");
            }
            @SuppressWarnings("unchecked")
            JobDetail jobDetail = JobBuilder.newJob((Class<? extends Job>) jobClass).withIdentity(jobName, jobGroupName)
                    .withDescription(description).storeDurably().requestRecovery().build();
            jobDetail.getJobDataMap().put("extraData", data);
            TriggerBuilder<Trigger> triggerBuilder = TriggerBuilder.newTrigger();
            triggerBuilder.withIdentity(triggerName, triggerGroupName);
            triggerBuilder.startNow();
            triggerBuilder.withDescription(description);
            triggerBuilder.withSchedule(CronScheduleBuilder.cronSchedule(cron));
            CronTrigger trigger = (CronTrigger) triggerBuilder.build();
            scheduler.scheduleJob(jobDetail, trigger);
            if (!scheduler.isShutdown()) {
                scheduler.start();
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }

    public void addJobs(QrtzTriggers qrtzTriggers1, QrtzTriggers qrtzTriggers2, Object... objects) {

        try {
            String jobClassName = qrtzTriggers1.getJobClassName();
            ClassLoader loader = ClassLoader.getSystemClassLoader();
            Class<?> loadedClass = Class.forName(jobClassName);
            if (!Job.class.isAssignableFrom(loadedClass)) {
                throw new IllegalArgumentException("The class " + jobClassName + " does not implement Job.");
            }
            @SuppressWarnings("unchecked")
            Class<? extends Job> jobClass = (Class<? extends Job>) loadedClass;
            String jobName = qrtzTriggers1.getJobName();
            String jobGroupName = qrtzTriggers1.getJobGroup();
            JobDetail jobDetail = JobBuilder.newJob(jobClass).withIdentity(jobName, jobGroupName).storeDurably()
                    .requestRecovery().build();
            List<QrtzTriggers> qrtzTriggersList = new ArrayList<>();
            qrtzTriggersList.add(qrtzTriggers1);
            qrtzTriggersList.add(qrtzTriggers2);
            List<CronTrigger> cronTriggers = new ArrayList<>();
            for (QrtzTriggers qrtzTrigger : qrtzTriggersList) {
                TriggerBuilder<Trigger> triggerBuilder = TriggerBuilder.newTrigger();
                triggerBuilder.withIdentity(qrtzTrigger.getTriggerName(), qrtzTrigger.getTriggerGroup());
                triggerBuilder.startNow();
                triggerBuilder.withDescription(qrtzTrigger.getDescription());
                triggerBuilder.withSchedule(CronScheduleBuilder.cronSchedule(qrtzTrigger.getCron()));
                triggerBuilder.forJob(jobDetail);
                CronTrigger trigger = (CronTrigger) triggerBuilder.build();
                cronTriggers.add(trigger);
            }
            if (objects != null) {
                for (int i = 0; i < objects.length; i++) {
                    // jobDetail.getJobDataMap().put("extraData", "admin");
                    // Get data by using: JobDataMap dataMap =
                    // context.getJobDetail().getJobDataMap();
                }
            }
            scheduler.addJob(jobDetail, true);
            scheduler.scheduleJob(cronTriggers.get(0));
            scheduler.scheduleJob(cronTriggers.get(1));
            if (!scheduler.isShutdown()) {
                scheduler.start();
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

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
            throw new RuntimeException(e);
        }
    }

    @LogMethod
    public void removeJob(String jobName, String jobGroupName, String triggerName, String triggerGroupName) {
        try {

            TriggerKey triggerKey = TriggerKey.triggerKey(triggerName, triggerGroupName);
            scheduler.pauseTrigger(triggerKey);
            scheduler.unscheduleJob(triggerKey);
            scheduler.deleteJob(JobKey.jobKey(jobName, jobGroupName));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public void pauseJob(String triggerName, String triggerGroupName) {
        try {
            TriggerKey triggerKey = TriggerKey.triggerKey(triggerName, triggerGroupName);
            scheduler.pauseTrigger(triggerKey);
        } catch (SchedulerException e) {
            e.printStackTrace();
        }
    }

    public void startJob(String triggerName, String triggerGroupName) {
        try {
            TriggerKey triggerKey = TriggerKey.triggerKey(triggerName, triggerGroupName);
            scheduler.resumeTrigger(triggerKey);
        } catch (SchedulerException e) {
            e.printStackTrace();
        }
    }

    public void startAllJobs() {
        try {
            scheduler.start();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public void shutdownAllJobs() {
        try {
            if (!scheduler.isShutdown()) {
                scheduler.shutdown();
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}