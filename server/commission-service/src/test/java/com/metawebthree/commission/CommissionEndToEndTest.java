package com.metawebthree.commission;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.springframework.boot.SpringApplication;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.web.client.RestTemplate;

import static org.assertj.core.api.Assertions.assertThat;

public class CommissionEndToEndTest extends PostgresTestBase {
    private static final int COMMISSION_PORT = 18088;
    private static final int USER_PORT = 18083;
    private static final int ORDER_PORT = 18084;
    private static final String COMMISSION_URL = "http://localhost:" + COMMISSION_PORT;
    private static final String USER_URL = "http://localhost:" + USER_PORT;
    private static final String ORDER_URL = "http://localhost:" + ORDER_PORT;

    private static ConfigurableApplicationContext commissionContext;
    private static ConfigurableApplicationContext userContext;
    private static ConfigurableApplicationContext orderContext;
    private static JdbcTemplate jdbcTemplate;
    private static RestTemplate restTemplate;

    @BeforeAll
    static void startServices() {
        jdbcTemplate = createJdbcTemplate();
        restTemplate = new RestTemplate();
        commissionContext = startCommissionService();
        userContext = startUserService();
        orderContext = startOrderService();
    }

    @AfterAll
    static void stopServices() {
        closeContext(orderContext);
        closeContext(userContext);
        closeContext(commissionContext);
    }

    @Test
    public void runsEndToEndFlow() {
        String referrerEmail = "referrer@demo.com";
        String buyerEmail = "buyer@demo.com";
        createUser(referrerEmail, null);
        Long referrerId = loadUserId(referrerEmail);
        createUser(buyerEmail, referrerId);
        Long buyerId = loadUserId(buyerEmail);
        assertRelationExists(buyerId, referrerId);
        Long orderId = createOrder(buyerId, new BigDecimal("100"));
        confirmReceive(buyerId, orderId);
        settleCommission();
        assertAvailableAmount(referrerId);
    }

    private static ConfigurableApplicationContext startCommissionService() {
        return startService(com.metawebthree.commission.CommissionServiceApplication.class,
                COMMISSION_PORT, "classpath:db/init.sql", null);
    }

    private static ConfigurableApplicationContext startUserService() {
        return startService(com.metawebthree.UserServiceApplication.class,
                USER_PORT, "classpath:db/init.sql", COMMISSION_URL);
    }

    private static ConfigurableApplicationContext startOrderService() {
        return startService(com.metawebthree.OrderServiceApplication.class,
                ORDER_PORT, "classpath:schema.sql", COMMISSION_URL);
    }

    private static ConfigurableApplicationContext startService(Class<?> application, int port,
            String schemaLocation, String commissionBaseUrl) {
        SpringApplication app = new SpringApplication(application);
        app.setDefaultProperties(buildProperties(port, schemaLocation, commissionBaseUrl));
        return app.run();
    }

    private static Map<String, Object> buildProperties(int port, String schemaLocation, String commissionBaseUrl) {
        Map<String, Object> props = baseProperties(port, schemaLocation);
        if (commissionBaseUrl != null) {
            props.put("commission.service.base-url", commissionBaseUrl);
            props.put("commission.return-window-days", "0");
        }
        return props;
    }

    private static Map<String, Object> baseProperties(int port, String schemaLocation) {
        Map<String, Object> props = new HashMap<>();
        props.put("server.port", port);
        props.put("spring.datasource.url", POSTGRES.getJdbcUrl());
        props.put("spring.datasource.username", POSTGRES.getUsername());
        props.put("spring.datasource.password", POSTGRES.getPassword());
        props.put("spring.datasource.driver-class-name", POSTGRES.getDriverClassName());
        props.put("spring.sql.init.mode", "always");
        props.put("spring.sql.init.continue-on-error", "true");
        props.put("spring.sql.init.schema-locations", schemaLocation);
        props.put("spring.cloud.zookeeper.discovery.enabled", "false");
        props.put("spring.cloud.zookeeper.connect-string", "");
        return props;
    }

    private void createUser(String email, Long referrerId) {
        Map<String, Object> body = userCreatePayload(email, referrerId);
        HttpEntity<Map<String, Object>> entity = jsonEntity(body, null);
        restTemplate.postForEntity(USER_URL + "/user/create", entity, String.class);
    }

    private Map<String, Object> userCreatePayload(String email, Long referrerId) {
        Map<String, Object> body = new HashMap<>();
        body.put("email", email);
        body.put("password", "password");
        body.put("typeId", 1);
        if (referrerId != null) {
            body.put("referrerId", referrerId);
        }
        return body;
    }

    private Long loadUserId(String email) {
        String sql = "select id from \"User\" where email = ?";
        return jdbcTemplate.queryForObject(sql, Long.class, email);
    }

    private void assertRelationExists(Long userId, Long parentId) {
        String sql = "select count(1) from commission_relation where user_id = ? and parent_user_id = ?";
        Integer count = jdbcTemplate.queryForObject(sql, Integer.class, userId, parentId);
        assertThat(count).isEqualTo(1);
    }

    private Long createOrder(Long userId, BigDecimal unitPrice) {
        Map<String, Object> request = orderCreatePayload(unitPrice);
        HttpEntity<Map<String, Object>> entity = jsonEntity(request, userId);
        Map<?, ?> response = restTemplate.postForObject(ORDER_URL + "/order/create", entity, Map.class);
        return Long.valueOf(String.valueOf(response.get("data")));
    }

    private Map<String, Object> orderCreatePayload(BigDecimal unitPrice) {
        Map<String, Object> item = new HashMap<>();
        item.put("productId", 1);
        item.put("productName", "Demo");
        item.put("skuId", 1);
        item.put("quantity", 1);
        item.put("unitPrice", unitPrice);
        Map<String, Object> request = new HashMap<>();
        request.put("remark", "test");
        request.put("items", java.util.List.of(item));
        return request;
    }

    private void confirmReceive(Long userId, Long orderId) {
        HttpEntity<Void> entity = jsonEntity(null, userId);
        restTemplate.postForEntity(ORDER_URL + "/order/confirm-receive/" + orderId, entity, String.class);
    }

    private void settleCommission() {
        Map<String, Object> body = new HashMap<>();
        body.put("executeBefore", LocalDateTime.now().plusDays(2));
        HttpEntity<Map<String, Object>> entity = jsonEntity(body, null);
        restTemplate.postForEntity(COMMISSION_URL + "/v1/commission/settle", entity, String.class);
    }

    private void assertAvailableAmount(Long userId) {
        String sql = "select available_amount from commission_main where user_id = ?";
        BigDecimal amount = jdbcTemplate.queryForObject(sql, BigDecimal.class, userId);
        assertThat(amount).isNotNull();
        assertThat(amount.compareTo(BigDecimal.ZERO)).isGreaterThan(0);
    }

    private static JdbcTemplate createJdbcTemplate() {
        return new JdbcTemplate(new org.springframework.jdbc.datasource.DriverManagerDataSource(
                POSTGRES.getJdbcUrl(), POSTGRES.getUsername(), POSTGRES.getPassword()));
    }

    private HttpEntity<Map<String, Object>> jsonEntity(Map<String, Object> body, Long userId) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        if (userId != null) {
            headers.add("X-User-ID", String.valueOf(userId));
        }
        return new HttpEntity<>(body, headers);
    }

    private static void closeContext(ConfigurableApplicationContext context) {
        if (context != null) {
            context.close();
        }
    }
}
