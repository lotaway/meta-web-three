package com.metawebthree.rma.infrastructure.sequence;

import com.metawebthree.rma.domain.RmaSequenceGenerator;
import org.springframework.stereotype.Component;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.concurrent.atomic.AtomicLong;

@Component
public class RmaSequenceGeneratorImpl implements RmaSequenceGenerator {

    private static final AtomicLong SEQ = new AtomicLong(1);
    private static final Object SEQ_LOCK = new Object();
    private static final String DATE_FORMAT = "yyyyMMdd";
    private static final String RMA_NO_PREFIX = "RMA";
    private static final String RMA_NO_FORMAT = "%06d";
    private static final int MAX_SEQUENCE = 1000000;

    private static String seqDate = LocalDate.now().format(DateTimeFormatter.ofPattern(DATE_FORMAT));

    @Override
    public String generateRmaNo() {
        String datePart = LocalDate.now().format(DateTimeFormatter.ofPattern(DATE_FORMAT));
        synchronized (SEQ_LOCK) {
            if (!datePart.equals(seqDate)) {
                SEQ.set(1);
                seqDate = datePart;
            }
            long seq = SEQ.getAndIncrement();
            return RMA_NO_PREFIX + datePart + String.format(RMA_NO_FORMAT, seq % MAX_SEQUENCE);
        }
    }
}
