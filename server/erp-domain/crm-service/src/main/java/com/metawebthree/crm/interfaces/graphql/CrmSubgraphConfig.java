package com.metawebthree.crm.interfaces.graphql;

import com.apollographql.federation.graphqljava.Federation;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.crm.application.query.CampaignQueryService;
import com.metawebthree.crm.application.query.ContactQueryService;
import com.metawebthree.crm.application.query.LeadQueryService;
import com.metawebthree.crm.application.query.OpportunityQueryService;
import com.metawebthree.crm.application.query.TicketQueryService;
import com.metawebthree.crm.domain.entity.*;
import com.metawebthree.crm.interfaces.graphql.dto.*;
import graphql.GraphQL;
import graphql.TypeResolutionEnvironment;
import graphql.schema.DataFetchingEnvironment;
import graphql.schema.GraphQLObjectType;
import graphql.schema.GraphQLSchema;
import graphql.schema.idl.*;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.ResourcePatternResolver;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

@Configuration
public class CrmSubgraphConfig {

    private final LeadQueryService leadQueryService;
    private final OpportunityQueryService opportunityQueryService;
    private final TicketQueryService ticketQueryService;
    private final CampaignQueryService campaignQueryService;
    private final ContactQueryService contactQueryService;
    private final ResourcePatternResolver resourceResolver;

    public CrmSubgraphConfig(LeadQueryService leadQueryService,
                             OpportunityQueryService opportunityQueryService,
                             TicketQueryService ticketQueryService,
                             CampaignQueryService campaignQueryService,
                             ContactQueryService contactQueryService,
                             ResourcePatternResolver resourceResolver) {
        this.leadQueryService = leadQueryService;
        this.opportunityQueryService = opportunityQueryService;
        this.ticketQueryService = ticketQueryService;
        this.campaignQueryService = campaignQueryService;
        this.contactQueryService = contactQueryService;
        this.resourceResolver = resourceResolver;
    }

    @Bean
    public GraphQL graphQL() throws IOException {
        TypeDefinitionRegistry typeRegistry = buildSchema();
        RuntimeWiring wiring = buildRuntimeWiring();
        GraphQLSchema schema = buildFederationSchema(typeRegistry, wiring);
        return GraphQL.newGraphQL(schema).build();
    }

    private TypeDefinitionRegistry buildSchema() throws IOException {
        Resource[] resources = resourceResolver.getResources("classpath:graphql/*.graphqls");
        StringBuilder sb = new StringBuilder();
        for (Resource r : resources) {
            sb.append(new String(r.getInputStream().readAllBytes(), StandardCharsets.UTF_8));
        }
        return new SchemaParser().parse(sb.toString());
    }

    private RuntimeWiring buildRuntimeWiring() {
        return RuntimeWiring.newRuntimeWiring()
                .type("Query", wiring -> wiring
                        .dataFetcher("lead", this::leadDataFetcher)
                        .dataFetcher("leads", this::leadsDataFetcher)
                        .dataFetcher("opportunity", this::opportunityDataFetcher)
                        .dataFetcher("opportunities", this::opportunitiesDataFetcher)
                        .dataFetcher("ticket", this::ticketDataFetcher)
                        .dataFetcher("tickets", this::ticketsDataFetcher)
                        .dataFetcher("campaign", this::campaignDataFetcher)
                        .dataFetcher("campaigns", this::campaignsDataFetcher)
                        .dataFetcher("contact", this::contactDataFetcher)
                        .dataFetcher("contacts", this::contactsDataFetcher)
                )
                .build();
    }

    private GraphQLSchema buildFederationSchema(TypeDefinitionRegistry registry, RuntimeWiring wiring) {
        return Federation.transform(registry, wiring)
                .fetchEntities(this::fetchEntity)
                .resolveEntityType(this::resolveEntityType)
                .build();
    }

    @SuppressWarnings("unchecked")
    private Object fetchEntity(DataFetchingEnvironment env) {
        List<Map<String, Object>> representations = env.getArgument("representations");
        if (representations == null || representations.isEmpty()) return null;
        Map<String, Object> representation = representations.get(0);
        String typeName = (String) representation.get("__typename");
        Object idObj = representation.get("id");
        Long id = idObj instanceof Number ? ((Number) idObj).longValue() : Long.valueOf(idObj.toString());
        return resolveRepresentation(typeName, id);
    }

    private Object resolveRepresentation(String typeName, Long id) {
        return switch (typeName) {
            case "Lead" -> nullableDto(leadQueryService.getById(id), LeadDTO::from);
            case "Opportunity" -> nullableDto(opportunityQueryService.getById(id), OpportunityDTO::from);
            case "Ticket" -> nullableDto(ticketQueryService.getById(id), TicketDTO::from);
            case "Campaign" -> nullableDto(campaignQueryService.getById(id), CampaignDTO::from);
            case "Contact" -> nullableDto(contactQueryService.getById(id), ContactDTO::from);
            default -> null;
        };
    }

    private <T, R> R nullableDto(T entity, java.util.function.Function<T, R> mapper) {
        return entity != null ? mapper.apply(entity) : null;
    }

    private GraphQLObjectType resolveEntityType(TypeResolutionEnvironment env) {
        Object src = env.getObject();
        if (src instanceof TypedNode node) {
            return env.getSchema().getObjectType(node.__typename());
        }
        if (src instanceof Map<?, ?> map) {
            return env.getSchema().getObjectType((String) map.get("__typename"));
        }
        return null;
    }

    private LeadDTO leadDataFetcher(DataFetchingEnvironment env) {
        return nullableDto(leadQueryService.getById(argId(env)), LeadDTO::from);
    }

    private Connection<LeadDTO> leadsDataFetcher(DataFetchingEnvironment env) {
        List<Lead> leads = resolveLeads(env);
        return pageResult(leads, LeadDTO::from, l -> l.getId().toString());
    }

    private OpportunityDTO opportunityDataFetcher(DataFetchingEnvironment env) {
        return nullableDto(opportunityQueryService.getById(argId(env)), OpportunityDTO::from);
    }

    private Connection<OpportunityDTO> opportunitiesDataFetcher(DataFetchingEnvironment env) {
        List<Opportunity> opportunities = resolveOpportunities(env);
        return pageResult(opportunities, OpportunityDTO::from, o -> o.getId().toString());
    }

    private TicketDTO ticketDataFetcher(DataFetchingEnvironment env) {
        return nullableDto(ticketQueryService.getById(argId(env)), TicketDTO::from);
    }

    private Connection<TicketDTO> ticketsDataFetcher(DataFetchingEnvironment env) {
        List<CustomerServiceTicket> tickets = resolveTickets(env);
        return pageResult(tickets, TicketDTO::from, t -> t.getId().toString());
    }

    private CampaignDTO campaignDataFetcher(DataFetchingEnvironment env) {
        return nullableDto(campaignQueryService.getById(argId(env)), CampaignDTO::from);
    }

    private Connection<CampaignDTO> campaignsDataFetcher(DataFetchingEnvironment env) {
        List<Campaign> campaigns = resolveCampaigns(env);
        return pageResult(campaigns, CampaignDTO::from, c -> c.getId().toString());
    }

    private ContactDTO contactDataFetcher(DataFetchingEnvironment env) {
        return nullableDto(contactQueryService.getById(argId(env)), ContactDTO::from);
    }

    private Connection<ContactDTO> contactsDataFetcher(DataFetchingEnvironment env) {
        List<Contact> contacts = resolveContacts(env);
        return pageResult(contacts, ContactDTO::from, c -> c.getId().toString());
    }

    private Long argId(DataFetchingEnvironment env) {
        Object id = env.getArgument("id");
        if (id == null) return null;
        if (id instanceof Number) return ((Number) id).longValue();
        return Long.valueOf(id.toString());
    }

    private List<Lead> resolveLeads(DataFetchingEnvironment env) {
        String keyword = env.getArgument("keyword");
        String status = env.getArgument("status");
        String source = env.getArgument("source");
        if (hasText(keyword)) return leadQueryService.search(keyword);
        if (hasText(status)) return leadQueryService.listByStatus(status);
        if (hasText(source)) return leadQueryService.listBySource(source);
        return leadQueryService.listAll();
    }

    private List<Opportunity> resolveOpportunities(DataFetchingEnvironment env) {
        String keyword = env.getArgument("keyword");
        String stage = env.getArgument("stage");
        if (hasText(keyword)) return opportunityQueryService.search(keyword);
        if (hasText(stage)) return opportunityQueryService.listByStage(stage);
        return opportunityQueryService.listAll();
    }

    private List<CustomerServiceTicket> resolveTickets(DataFetchingEnvironment env) {
        String priority = env.getArgument("priority");
        String status = env.getArgument("status");
        if (hasText(priority)) return ticketQueryService.listByPriority(priority);
        if (hasText(status)) return ticketQueryService.listByStatus(status);
        return ticketQueryService.listAll();
    }

    private List<Campaign> resolveCampaigns(DataFetchingEnvironment env) {
        String status = env.getArgument("status");
        String type = env.getArgument("type");
        if (hasText(status)) return campaignQueryService.listByStatus(status);
        if (hasText(type)) return campaignQueryService.listByType(type);
        return campaignQueryService.listAll();
    }

    private List<Contact> resolveContacts(DataFetchingEnvironment env) {
        String customerId = env.getArgument("customerId");
        String keyword = env.getArgument("keyword");
        if (hasText(customerId)) return contactQueryService.listByCustomerId(Long.valueOf(customerId));
        if (hasText(keyword)) return List.of();
        return contactQueryService.listAll();
    }

    private static boolean hasText(String s) {
        return s != null && !s.isBlank();
    }

    private <T, R> Connection<R> pageResult(List<T> items,
            java.util.function.Function<T, R> mapper,
            java.util.function.Function<T, String> cursorFn) {
        List<Edge<R>> edges = items.stream()
                .map(item -> new Edge<>(cursorFn.apply(item), mapper.apply(item)))
                .toList();
        return new Connection<>(edges, items.size(), new PageInfo(false, null));
    }
}
