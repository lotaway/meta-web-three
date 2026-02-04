package com.metawebthree.commission;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import com.metawebthree.commission.application.CommissionCommandService;
import com.metawebthree.commission.application.CommissionQueryService;
import com.metawebthree.commission.domain.CommissionAccount;
import com.metawebthree.commission.domain.CommissionRecord;
import com.metawebthree.commission.domain.CommissionRecordStatus;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest(
    classes = CommissionServiceApplication.class,
    properties = {
        "spring.sql.init.mode=always",
        "spring.sql.init.schema-locations=classpath:db/init.sql",
        "commission.buy-rate=0.10",
        "commission.level-rates=0.4,0.2",
        "commission.max-levels=2",
        "commission.return-window-days=0"
    }
)
public class CommissionCommandServiceTest extends PostgresTestBase {

    @Autowired
    private CommissionCommandService commandService;

    @Autowired
    private CommissionQueryService queryService;

    @Test
    public void calculatesAndSettlesCommission() {
        Long referrerId = 1001L;
        Long buyerId = 2001L;
        LocalDateTime availableAt = LocalDateTime.now();
        createRelationAndCommission(referrerId, buyerId, 3001L, "100", availableAt);
        assertPending(referrerId);
        commandService.settleBefore(availableAt.plusMinutes(1));
        assertAvailable(referrerId);
        assertAvailableAmount(referrerId, "4.0");
    }

    @Test
    public void cancelsPendingCommission() {
        Long referrerId = 1002L;
        Long buyerId = 2002L;
        LocalDateTime availableAt = LocalDateTime.now().plusDays(1);
        createRelationAndCommission(referrerId, buyerId, 3002L, "200", availableAt);
        commandService.cancelByOrderId(3002L);
        assertCanceled(referrerId);
        assertTotalAmount(referrerId, "0");
    }

    private void createRelationAndCommission(Long referrerId, Long buyerId, Long orderId,
            String amount, LocalDateTime availableAt) {
        commandService.bindRelation(buyerId, referrerId);
        commandService.calculateForOrder(orderId, buyerId, new BigDecimal(amount), availableAt);
    }

    private CommissionRecord loadFirstRecord(Long userId) {
        List<CommissionRecord> records = queryService.listRecords(userId, null, 1, 10);
        return records.get(0);
    }

    private void assertPending(Long userId) {
        assertThat(loadFirstRecord(userId).getStatus())
                .isEqualTo(CommissionRecordStatus.PENDING.name());
    }

    private void assertAvailable(Long userId) {
        assertThat(loadFirstRecord(userId).getStatus())
                .isEqualTo(CommissionRecordStatus.AVAILABLE.name());
    }

    private void assertCanceled(Long userId) {
        assertThat(loadFirstRecord(userId).getStatus())
                .isEqualTo(CommissionRecordStatus.CANCELED.name());
    }

    private void assertAvailableAmount(Long userId, String expected) {
        CommissionAccount account = queryService.getAccount(userId);
        assertThat(account.getAvailableAmount()).isEqualByComparingTo(expected);
    }

    private void assertTotalAmount(Long userId, String expected) {
        CommissionAccount account = queryService.getAccount(userId);
        assertThat(account.getTotalAmount()).isEqualByComparingTo(expected);
    }
}
