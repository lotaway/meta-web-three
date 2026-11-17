package com.metawebthree.common.autoconfigure;

import org.quartz.Scheduler;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.context.annotation.Import;

import com.metawebthree.common.services.QuartzManager;

/**
 * Registers the shared {@link QuartzManager} helper once Quartz is on the
 * classpath. The manager only holds an {@code ObjectProvider<Scheduler>}, so
 * it is safe to register even for services that do not schedule any job.
 * Per-service job registration stays behind each service's own
 * {@code @ConditionalOnProperty} configuration.
 */
@AutoConfiguration
@ConditionalOnClass(Scheduler.class)
@Import(QuartzManager.class)
public class QuartzAutoConfiguration {
}
