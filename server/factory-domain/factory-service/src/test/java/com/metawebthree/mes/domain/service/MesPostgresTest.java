package com.metawebthree.mes.domain.service;

import com.baomidou.mybatisplus.extension.spring.MybatisSqlSessionFactoryBean;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.mes.adapter.http.controller.*;
import com.metawebthree.mes.domain.entity.*;
import com.metawebthree.mes.domain.repository.*;
import com.metawebthree.mes.infrastructure.persistence.repository.*;
import org.apache.ibatis.session.SqlSessionFactory;
import org.mybatis.spring.annotation.MapperScan;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.springframework.context.annotation.*;
import org.springframework.core.io.ClassPathResource;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.datasource.DriverManagerDataSource;
import org.springframework.jdbc.datasource.DataSourceTransactionManager;
import org.springframework.jdbc.datasource.init.ScriptUtils;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.annotation.EnableTransactionManagement;
import org.springframework.transaction.support.TransactionTemplate;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.http.converter.json.MappingJackson2HttpMessageConverter;
import javax.sql.DataSource;
import java.sql.Connection;
import java.util.*;
import java.util.concurrent.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

/** Opt-in real database tests; each run creates and retains an isolated schema. */
@EnabledIfEnvironmentVariable(named="MES_TEST_URL",matches=".+")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class MesPostgresTest {
    static String schema;
    AnnotationConfigApplicationContext context;
    JdbcTemplate jdbc;
    ObjectMapper json;
    MockMvc mvc;
    @Configuration
    @EnableTransactionManagement(proxyTargetClass=true)
    @MapperScan("com.metawebthree.mes.infrastructure.persistence.mapper")
    @Import({AndonTypeRepositoryImpl.class,AndonLevelRepositoryImpl.class,AndonEventRepositoryImpl.class,
        EquipmentTypeRepositoryImpl.class,EquipmentStatusConfigRepositoryImpl.class,EquipmentStatusTransitionRepositoryImpl.class,
        WorkOrderSplitRuleRepositoryImpl.class,WorkOrderRepositoryImpl.class,EquipmentRepositoryImpl.class,
        StatusMachineRepositoryImpl.class,WorkOrderSplitServiceImpl.class,AndonServiceImpl.class,
        EquipmentTypeService.class,EquipmentStatusService.class,EquipmentTransitionPolicy.class,
        WorkOrderSplitController.class,AndonController.class,EquipmentConfigurationController.class,EquipmentController.class})
    static class Config {
        @Bean DataSource dataSource() {
            return new DriverManagerDataSource(System.getenv("MES_TEST_URL")+"?currentSchema="+schema,
                    System.getenv("MES_TEST_USER"),System.getenv("MES_TEST_PASSWORD"));
        }
        @Bean ObjectMapper objectMapper() { return new ObjectMapper().findAndRegisterModules(); }
        @Bean SqlSessionFactory sqlSessionFactory(DataSource ds) throws Exception {
            MybatisSqlSessionFactoryBean factory=new MybatisSqlSessionFactoryBean(); factory.setDataSource(ds);
            com.baomidou.mybatisplus.core.MybatisConfiguration configuration=new com.baomidou.mybatisplus.core.MybatisConfiguration();
            configuration.setMapUnderscoreToCamelCase(true); factory.setConfiguration(configuration); return factory.getObject();
        }
        @Bean PlatformTransactionManager transactionManager(DataSource ds) { return new DataSourceTransactionManager(ds); }
    }
    @BeforeAll void setup() throws Exception {
        schema="mes_verify_"+UUID.randomUUID().toString().replace("-","");
        DataSource admin=new DriverManagerDataSource(System.getenv("MES_TEST_URL"),System.getenv("MES_TEST_USER"),System.getenv("MES_TEST_PASSWORD"));
        new JdbcTemplate(admin).execute("CREATE SCHEMA "+schema);
        context=new AnnotationConfigApplicationContext(Config.class);
        jdbc=new JdbcTemplate(context.getBean(DataSource.class)); json=context.getBean(ObjectMapper.class);
        try(Connection connection=context.getBean(DataSource.class).getConnection()) {
            for(int i=0;i<2;i++) ScriptUtils.executeSqlScript(connection,new ClassPathResource("db/migration/V20261015__mes_configuration_and_safe_split.sql"));
        }
        mvc=MockMvcBuilders.standaloneSetup(context.getBean(AndonController.class),context.getBean(EquipmentConfigurationController.class),
                context.getBean(EquipmentController.class),context.getBean(WorkOrderSplitController.class))
            .setControllerAdvice(new MesOperationExceptionHandler()).setMessageConverters(new MappingJackson2HttpMessageConverter(json)).build();
        System.out.println("MES_TEST_SCHEMA="+schema);
    }
    @AfterAll void finish() { if(context!=null) context.close(); }
    String code() { return UUID.randomUUID().toString(); }
    <T> T bean(Class<T> type) { return context.getBean(type); }
    WorkOrder parent() {
        WorkOrder p=new WorkOrder(); p.createWithType("WO-"+code(),"P","Product",100,"W","R","SPECIAL"); return bean(WorkOrderRepository.class).save(p);
    }
    WorkOrderSplitRule rule(String type) {
        WorkOrderSplitRule r=new WorkOrderSplitRule(); r.create("Rule",code(),type); r.setSplitQuantity(30); return bean(WorkOrderSplitRuleRepository.class).save(r);
    }
    @Test void persistentRepositoriesAndJsonRoundTrip() {
        AndonType t=bean(AndonService.class).createAndonType(code(),"Fault",AndonType.AndonCategory.EQUIPMENT);
        assertNotNull(t.getId()); assertNotNull(bean(AndonTypeRepository.class).findById(t.getId()).orElseThrow().getCreatedAt());
        assertTrue(bean(AndonTypeRepository.class).findByCategory(t.getCategory()).stream().anyMatch(x -> x.getId().equals(t.getId())));
        assertThrows(Exception.class,() -> bean(AndonService.class).createAndonType(t.getTypeCode(),"Duplicate",t.getCategory()));
        AndonLevel l=bean(AndonService.class).createAndonLevel(code(),"High",Math.abs(code().hashCode())+1);
        assertEquals(l.getId(),bean(AndonLevelRepository.class).findByLevelValue(l.getLevelValue()).orElseThrow().getId());
        EquipmentType et=bean(EquipmentTypeService.class).createType(code(),"Machine","TEST");
        EquipmentType.EquipmentTypeAttribute a=new EquipmentType.EquipmentTypeAttribute(); a.create("power","Power","DECIMAL"); et.addAttribute(a); bean(EquipmentTypeRepository.class).update(et);
        assertEquals("power",bean(EquipmentTypeRepository.class).findById(et.getId()).orElseThrow().getAttributes().get(0).getAttributeCode());
        WorkOrderSplitRule r=rule("MANUAL"); WorkOrderSplitRule.SplitCondition c=new WorkOrderSplitRule.SplitCondition(); c.setField("quantity"); c.setOperator("GTE"); c.setValue(10L); r.setConditions(List.of(c)); bean(WorkOrderSplitRuleRepository.class).save(r);
        assertEquals("GTE",bean(WorkOrderSplitRuleRepository.class).findById(r.getId()).orElseThrow().getConditions().get(0).getOperator());
        AndonEvent event=bean(AndonService.class).triggerAndon(t.getId(),l.getId(),"API_CALL","W",null,"E","U","Operator","Fault");
        assertEquals(event.getId(),bean(AndonEventRepository.class).findByEventNo(event.getEventNo()).orElseThrow().getId());
        assertFalse(bean(AndonEventRepository.class).findByWorkshopId("W").isEmpty());
        // Recreate repository/context to prove durable reload rather than memory identity.
        try(AnnotationConfigApplicationContext reloaded=new AnnotationConfigApplicationContext(Config.class)) {
            assertEquals(event.getEventNo(),reloaded.getBean(AndonEventRepository.class).findById(event.getId()).orElseThrow().getEventNo());
        }
        l.deactivate(); bean(AndonLevelRepository.class).update(l);
        assertThrows(IllegalStateException.class,() -> bean(AndonService.class).triggerAndon(t.getId(),l.getId(),"API_CALL","W",null,null,"U","Name","Fault"));
    }
    @Test void splitPersistsAndRepeatRejects() {
        WorkOrder p=parent(); WorkOrderSplitRule r=rule("BY_QUANTITY");
        List<WorkOrder> children=bean(WorkOrderSplitService.class).splitWorkOrder(p,r.getId());
        assertEquals(List.of(30,30,30,10),children.stream().map(WorkOrder::getQuantity).toList());
        WorkOrder stored=bean(WorkOrderRepository.class).findById(p.getId()).orElseThrow();
        assertEquals(100,stored.getQuantity()); assertEquals(WorkOrder.WorkOrderStatus.CANCELLED,stored.getStatus());
        assertEquals(4,bean(WorkOrderSplitService.class).getChildWorkOrders(p.getId()).size());
        assertThrows(IllegalStateException.class,() -> bean(WorkOrderSplitService.class).splitWorkOrder(p,r.getId()));
    }
    @Test void splitRollbackOnSecondChildFailure() {
        WorkOrder p=parent(); WorkOrderSplitRule r=rule("BY_QUANTITY");
        // Existing conflicting number forces failure after the first child insert.
        WorkOrder conflict=new WorkOrder(); conflict.create(p.getWorkOrderNo()+"-S02","P","Product",1,"W","R"); bean(WorkOrderRepository.class).save(conflict);
        assertThrows(Exception.class,() -> bean(WorkOrderSplitService.class).splitWorkOrder(p,r.getId()));
        assertTrue(bean(WorkOrderSplitService.class).getChildWorkOrders(p.getId()).isEmpty());
        assertEquals(WorkOrder.WorkOrderStatus.DRAFT,bean(WorkOrderRepository.class).findById(p.getId()).orElseThrow().getStatus());
    }
    @Test void concurrentSplitOnlyOnce() throws Exception {
        WorkOrder p=parent(); WorkOrderSplitRule r=rule("BY_QUANTITY"); ExecutorService pool=Executors.newFixedThreadPool(2);
        CountDownLatch start=new CountDownLatch(1);
        Callable<Boolean> call=() -> {start.await(); try {bean(WorkOrderSplitService.class).splitWorkOrder(p,r.getId()); return true;} catch(IllegalStateException e) {return false;}};
        try {Future<Boolean> a=pool.submit(call),b=pool.submit(call); start.countDown(); assertNotEquals(a.get(15,TimeUnit.SECONDS),b.get(15,TimeUnit.SECONDS));}
        finally {pool.shutdown();}
        assertEquals(4,bean(WorkOrderSplitService.class).getChildWorkOrders(p.getId()).size());
    }
    @Test void equipmentHttpAndRuleGuards() throws Exception {
        String response=mvc.perform(post("/api/mes/equipment-types").contentType("application/json")
                .content(json.writeValueAsString(Map.of("typeCode",code(),"typeName","Machine","category","TEST"))))
                .andExpect(status().isOk()).andReturn().getResponse().getContentAsString();
        long typeId=json.readTree(response).get("id").asLong();
        for(String statusCode:List.of("IDLE","RUNNING")) {
            mvc.perform(post("/api/mes/equipment-types/"+typeId+"/statuses").contentType("application/json")
                    .content(json.writeValueAsString(Map.of("statusCode",statusCode,"statusName",statusCode,"statusCategory","TEST","initial",statusCode.equals("IDLE")))))
                    .andExpect(status().isOk());
        }
        mvc.perform(post("/api/mes/equipment-types/"+typeId+"/transitions").contentType("application/json")
                .content("{\"fromStatus\":\"IDLE\",\"toStatus\":\"RUNNING\",\"action\":\"MANUAL\"}")).andExpect(status().isOk());
        String equipment=mvc.perform(post("/api/mes/equipment").contentType("application/json")
                .content(json.writeValueAsString(Map.of("equipmentCode",code(),"equipmentName","Machine","equipmentTypeId",typeId,"workshopId","W"))))
                .andExpect(status().isOk()).andReturn().getResponse().getContentAsString();
        long id=json.readTree(equipment).get("id").asLong();
        Equipment stored=bean(EquipmentRepository.class).findById(id).orElseThrow();
        assertEquals(Equipment.EquipmentStatus.IDLE,stored.getStatus());
        EquipmentStatusTransition transition=bean(EquipmentStatusTransitionRepository.class)
                .findByEquipmentTypeIdAndFromStatusCodeAndToStatusCode(typeId,"IDLE","RUNNING").orElseThrow();
        transition.setCondition("temperature < 80"); bean(EquipmentStatusTransitionRepository.class).update(transition);
        assertFalse(bean(EquipmentTransitionPolicy.class).allows(stored,"RUNNING"));
        mvc.perform(post("/api/mes/equipment/"+id+"/start-task").contentType("application/json").content("{\"taskNo\":\"TASK\"}"))
                .andExpect(status().isConflict());
        transition.setCondition(null); bean(EquipmentStatusTransitionRepository.class).update(transition);
        assertNull(bean(EquipmentStatusTransitionRepository.class).findById(transition.getId()).orElseThrow().getConditionExpression());
        EquipmentStatusConfig source=bean(EquipmentStatusConfigRepository.class).findByEquipmentTypeIdAndStatusCode(typeId,"IDLE").orElseThrow();
        source.setIsFinal(true); bean(EquipmentStatusConfigRepository.class).update(source);
        assertFalse(bean(EquipmentTransitionPolicy.class).allows(stored,"RUNNING"));
        source.setIsFinal(false); bean(EquipmentStatusConfigRepository.class).update(source);
        EquipmentStatusConfig target=bean(EquipmentStatusConfigRepository.class).findByEquipmentTypeIdAndStatusCode(typeId,"RUNNING").orElseThrow();
        target.deactivate(); bean(EquipmentStatusConfigRepository.class).update(target);
        assertFalse(bean(EquipmentTransitionPolicy.class).allows(stored,"RUNNING"));
        target.activate(); bean(EquipmentStatusConfigRepository.class).update(target);
        mvc.perform(post("/api/mes/equipment/"+id+"/transition").contentType("application/json").content("{\"targetStatus\":\"RUNNING\"}"))
                .andExpect(status().isOk());
        assertEquals(Equipment.EquipmentStatus.RUNNING,bean(EquipmentRepository.class).findById(id).orElseThrow().getStatus());
        assertFalse(bean(EquipmentStatusConfigRepository.class).findByEquipmentTypeIdAndIsActive(typeId,true).isEmpty());
        assertThrows(Exception.class,() -> bean(EquipmentTypeService.class).addStatusTransition(typeId,"RUNNING","MISSING","MANUAL"));
    }

    @Test void genericMachineOverridesDedicatedRules() {
        EquipmentType type=bean(EquipmentTypeService.class).createType(code(),"Machine","TEST");
        bean(EquipmentTypeService.class).addStatusConfig(type.getId(),"IDLE","Idle","TEST");
        bean(EquipmentTypeService.class).addStatusConfig(type.getId(),"RUNNING","Running","TEST");
        bean(EquipmentTypeService.class).addStatusTransition(type.getId(),"IDLE","RUNNING","MANUAL");
        long machineId=jdbc.queryForObject("INSERT INTO mes_status_machine(machine_code,entity_type,status) VALUES (?, 'EQUIPMENT','ACTIVE') RETURNING id",Long.class,code());
        for(String s:List.of("IDLE","RUNNING")) jdbc.update("INSERT INTO mes_status_config(machine_id,status_code,is_initial,is_final) VALUES (?,?,?,false)",machineId,s,s.equals("IDLE"));
        type.bindStatusMachine(machineId); bean(EquipmentTypeRepository.class).update(type);
        Equipment equipment=new Equipment(); equipment.create("E","Machine",type.getId(),type.getTypeCode(),"W");
        assertFalse(bean(EquipmentTransitionPolicy.class).allows(equipment,"RUNNING"));
        jdbc.update("INSERT INTO mes_status_transition(machine_id,from_status,to_status,transition_action,is_auto_transition) VALUES (?,'IDLE','RUNNING','MANUAL',false)",machineId);
        assertTrue(bean(EquipmentTransitionPolicy.class).allows(equipment,"RUNNING"));
        jdbc.update("UPDATE mes_status_transition SET condition_expression='guard' WHERE machine_id=?",machineId);
        assertFalse(bean(EquipmentTransitionPolicy.class).allows(equipment,"RUNNING"));
    }

    @Test void manualSplitHttpAndUnsupportedStrategy() throws Exception {
        WorkOrder parent=parent(); WorkOrderSplitRule manual=rule("MANUAL");
        mvc.perform(post("/api/mes/work-order-splits/"+parent.getId()).contentType("application/json")
                .content(json.writeValueAsString(Map.of("ruleId",manual.getId(),"quantities",List.of(40,60)))))
                .andExpect(status().isOk()).andExpect(jsonPath("$[0].quantity").value(40));
        mvc.perform(get("/api/mes/work-order-splits/"+parent.getId()+"/children")).andExpect(jsonPath("$.length()").value(2));
        WorkOrder other=parent(); WorkOrderSplitRule bom=rule("BY_BOM");
        mvc.perform(post("/api/mes/work-order-splits/"+other.getId()).contentType("application/json")
                .content(json.writeValueAsString(Map.of("ruleId",bom.getId())))).andExpect(status().isUnprocessableEntity());
        assertTrue(bean(WorkOrderRepository.class).findByParentWorkOrderId(other.getId()).isEmpty());
        mvc.perform(post("/api/mes/work-order-splits/rules").contentType("application/json")
                .content("{\"ruleCode\":\"invalid\",\"ruleName\":\"Invalid\",\"splitType\":\"BY_PROCESS\"}"))
                .andExpect(status().isBadRequest());
    }

    @Test void andonHttpLifecycleAndInvalidOrder() throws Exception {
        String type=mvc.perform(post("/api/mes/andon/types").contentType("application/json").content(json.writeValueAsString(Map.of("typeCode",code(),"typeName","Fault","category","EQUIPMENT")))).andExpect(status().isOk()).andReturn().getResponse().getContentAsString();
        String level=mvc.perform(post("/api/mes/andon/levels").contentType("application/json").content(json.writeValueAsString(Map.of("levelCode",code(),"levelName","Urgent","levelValue",Math.abs(code().hashCode())+1)))).andExpect(status().isOk()).andReturn().getResponse().getContentAsString();
        String event=mvc.perform(post("/api/mes/andon/events").contentType("application/json").content(json.writeValueAsString(Map.of("andonTypeId",json.readTree(type).get("id").asLong(),"andonLevelId",json.readTree(level).get("id").asLong(),"triggerMethod","API_CALL","workshopId","W","reporterId","U","reporterName","Operator","description","Fault")))).andExpect(status().isOk()).andReturn().getResponse().getContentAsString();
        long id=json.readTree(event).get("id").asLong();
        mvc.perform(post("/api/mes/andon/events/"+id+"/close")).andExpect(status().isConflict());
        mvc.perform(post("/api/mes/andon/events/"+id+"/acknowledge").contentType("application/json").content("{\"handlerId\":\"H\",\"handlerName\":\"Handler\"}")).andExpect(status().isOk());
        for(String action:List.of("handle","resolve","close")) mvc.perform(post("/api/mes/andon/events/"+id+"/"+action)).andExpect(status().isOk());
        mvc.perform(get("/api/mes/andon/events/"+id)).andExpect(jsonPath("$.status").value("CLOSED"));
        mvc.perform(post("/api/mes/andon/types").contentType("application/json").content("{}" )).andExpect(status().isBadRequest());
    }
}
