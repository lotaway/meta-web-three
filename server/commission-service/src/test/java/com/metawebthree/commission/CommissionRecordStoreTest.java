package com.metawebthree.commission;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import com.metawebthree.commission.domain.ports.CommissionRecordStore;
import com.metawebthree.commission.domain.CommissionRecord;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest(
    classes = CommissionServiceApplication.class,
    properties = {
        "spring.sql.init.mode=always",
        "spring.sql.init.schema-locations=classpath:db/init.sql"
    }
)
public class CommissionRecordStoreTest extends PostgresTestBase {

    @Autowired
    private CommissionRecordStore recordStore;

    @Test
    public void storesAndLoadsRecords() {
        CommissionRecord record = new CommissionRecord();
        record.setOrderId(9001L);
        record.setUserId(8001L);
        record.setFromUserId(7001L);
        record.setLevel(1);
        record.setAmount(new BigDecimal("5.5"));
        record.setStatus("PENDING");
        record.setAvailableAt(LocalDateTime.now());
        record.setCreatedAt(LocalDateTime.now());
        record.setUpdatedAt(LocalDateTime.now());
        recordStore.save(record);
        List<CommissionRecord> records = recordStore.findByUserId(8001L, null, 1, 10);
        assertThat(records).isNotEmpty();
        assertThat(records.get(0).getOrderId()).isEqualTo(9001L);
    }
}
