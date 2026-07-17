package com.metawebthree.dom.infrastructure.service;

import com.metawebthree.dom.domain.service.DomSequenceGenerator;
import org.springframework.stereotype.Component;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.concurrent.atomic.AtomicLong;

@Component
public class DomSequenceGeneratorImpl implements DomSequenceGenerator {

    private final AtomicLong seqCounter = new AtomicLong(System.currentTimeMillis() % 1000000);

    @Override
    public String generateDomOrderNo() {
        String datePart = LocalDate.now().format(DateTimeFormatter.ofPattern("yyyyMMdd"));
        long seq = seqCounter.incrementAndGet() % 1000000;
        return "DOM" + datePart + String.format("%06d", seq);
    }
}
