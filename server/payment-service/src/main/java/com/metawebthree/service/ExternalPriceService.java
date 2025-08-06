package com.metawebthree.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.entity.CryptoPrice;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.concurrent.TimeUnit;

/**
 * 外部价格服务
 *
 * TODO: 如需接入自定义汇率/价格源，请实现 fetchFromSource、buildApiUrl、parsePriceResponse 等方法，
 * 并在 fetchPrice 方法中添加自定义 source。
 * 推荐将第三方API调用、签名、认证等逻辑封装为独立方法或类，便于后续维护和切换。
 *
 * 示例：
 * 1. 新增 source 名称，如 "myexchanger"
 * 2. 在 buildApiUrl 增加对应API地址
 * 3. 在 parsePriceResponse 增加解析逻辑
 * 4. 在 fetchPrice 的 sources 数组中添加新 source
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class ExternalPriceService {
    
    private final OkHttpClient httpClient;
    private final ObjectMapper objectMapper;
    
    /**
     * 从多个交易所获取价格
     */
    public CryptoPrice fetchPrice(String symbol) {
        String[] sources = {"binance", "coinbase", "okx"};
        
        for (String source : sources) {
            try {
                CryptoPrice price = fetchFromSource(symbol, source);
                if (price != null) {
                    return price;
                }
            } catch (Exception e) {
                log.warn("Failed to fetch price from {} for {}: {}", source, symbol, e.getMessage());
            }
        }
        
        throw new RuntimeException("Failed to fetch price from all sources for " + symbol);
    }
    
    /**
     * 从指定交易所获取价格
     */
    private CryptoPrice fetchFromSource(String symbol, String source) throws Exception {
        String url = buildApiUrl(symbol, source);
        Request request = new Request.Builder().url(url).build();
        
        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new RuntimeException("HTTP " + response.code() + " for " + url);
            }
            
            String responseBody = response.body().string();
            return parsePriceResponse(responseBody, symbol, source);
        }
    }
    
    /**
     * 构建API URL
     *
     * TODO: 如需接入新的价格源，请在此处添加对应的API地址和参数拼接逻辑。
     */
    private String buildApiUrl(String symbol, String source) {
        String baseCurrency = symbol.split("-")[0];
        String quoteCurrency = symbol.split("-")[1];
        
        switch (source) {
            case "binance":
                return String.format("https://api.binance.com/api/v3/ticker/price?symbol=%s%s", 
                    baseCurrency, quoteCurrency);
            case "coinbase":
                return String.format("https://api.coinbase.com/v2/prices/%s-%s/spot", 
                    baseCurrency, quoteCurrency);
            case "okx":
                return String.format("https://www.okx.com/api/v5/market/ticker?instId=%s-%s", 
                    baseCurrency, quoteCurrency);
            default:
                throw new IllegalArgumentException("Unsupported source: " + source);
        }
    }
    
    /**
     * 解析价格响应
     *
     * TODO: 如需接入新的价格源，请在此处添加对应的响应解析逻辑。
     */
    private CryptoPrice parsePriceResponse(String responseBody, String symbol, String source) throws Exception {
        JsonNode root = objectMapper.readTree(responseBody);
        String baseCurrency = symbol.split("-")[0];
        String quoteCurrency = symbol.split("-")[1];
        
        BigDecimal price = BigDecimal.ZERO;
        BigDecimal bidPrice = BigDecimal.ZERO;
        BigDecimal askPrice = BigDecimal.ZERO;
        BigDecimal volume24h = BigDecimal.ZERO;
        BigDecimal change24h = BigDecimal.ZERO;
        BigDecimal changePercent24h = BigDecimal.ZERO;
        
        switch (source) {
            case "binance":
                price = new BigDecimal(root.get("price").asText());
                // Binance API只返回价格，其他字段需要额外调用
                break;
                
            case "coinbase":
                price = new BigDecimal(root.get("data").get("amount").asText());
                break;
                
            case "okx":
                JsonNode data = root.get("data").get(0);
                price = new BigDecimal(data.get("last").asText());
                bidPrice = new BigDecimal(data.get("bidPx").asText());
                askPrice = new BigDecimal(data.get("askPx").asText());
                volume24h = new BigDecimal(data.get("vol24h").asText());
                change24h = new BigDecimal(data.get("change24h").asText());
                changePercent24h = new BigDecimal(data.get("changeRate").asText());
                break;
        }
        
        return CryptoPrice.builder()
                .symbol(symbol)
                .baseCurrency(baseCurrency)
                .quoteCurrency(quoteCurrency)
                .price(price)
                .bidPrice(bidPrice)
                .askPrice(askPrice)
                .volume24h(volume24h)
                .change24h(change24h)
                .changePercent24h(changePercent24h)
                .source(source)
                .timestamp(LocalDateTime.now())
                .build();
    }
    
    /**
     * 获取24小时价格变化
     */
    public CryptoPrice fetch24hTicker(String symbol, String source) throws Exception {
        String baseCurrency = symbol.split("-")[0];
        String quoteCurrency = symbol.split("-")[1];
        
        String url = "";
        switch (source) {
            case "binance":
                url = String.format("https://api.binance.com/api/v3/ticker/24hr?symbol=%s%s", 
                    baseCurrency, quoteCurrency);
                break;
            case "okx":
                url = String.format("https://www.okx.com/api/v5/market/ticker?instId=%s-%s", 
                    baseCurrency, quoteCurrency);
                break;
            default:
                throw new IllegalArgumentException("Unsupported source for 24h ticker: " + source);
        }
        
        Request request = new Request.Builder().url(url).build();
        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new RuntimeException("HTTP " + response.code() + " for " + url);
            }
            
            String responseBody = response.body().string();
            return parse24hTickerResponse(responseBody, symbol, source);
        }
    }
    
    /**
     * 解析24小时价格变化响应
     */
    private CryptoPrice parse24hTickerResponse(String responseBody, String symbol, String source) throws Exception {
        JsonNode root = objectMapper.readTree(responseBody);
        String baseCurrency = symbol.split("-")[0];
        String quoteCurrency = symbol.split("-")[1];
        
        BigDecimal price = BigDecimal.ZERO;
        BigDecimal bidPrice = BigDecimal.ZERO;
        BigDecimal askPrice = BigDecimal.ZERO;
        BigDecimal volume24h = BigDecimal.ZERO;
        BigDecimal change24h = BigDecimal.ZERO;
        BigDecimal changePercent24h = BigDecimal.ZERO;
        
        switch (source) {
            case "binance":
                JsonNode data = root;
                price = new BigDecimal(data.get("lastPrice").asText());
                bidPrice = new BigDecimal(data.get("bidPrice").asText());
                askPrice = new BigDecimal(data.get("askPrice").asText());
                volume24h = new BigDecimal(data.get("volume").asText());
                change24h = new BigDecimal(data.get("priceChange").asText());
                changePercent24h = new BigDecimal(data.get("priceChangePercent").asText());
                break;
                
            case "okx":
                JsonNode dataNode = root.get("data").get(0);
                price = new BigDecimal(dataNode.get("last").asText());
                bidPrice = new BigDecimal(dataNode.get("bidPx").asText());
                askPrice = new BigDecimal(dataNode.get("askPx").asText());
                volume24h = new BigDecimal(dataNode.get("vol24h").asText());
                change24h = new BigDecimal(dataNode.get("change24h").asText());
                changePercent24h = new BigDecimal(dataNode.get("changeRate").asText());
                break;
        }
        
        return CryptoPrice.builder()
                .symbol(symbol)
                .baseCurrency(baseCurrency)
                .quoteCurrency(quoteCurrency)
                .price(price)
                .bidPrice(bidPrice)
                .askPrice(askPrice)
                .volume24h(volume24h)
                .change24h(change24h)
                .changePercent24h(changePercent24h)
                .source(source)
                .timestamp(LocalDateTime.now())
                .build();
    }
} 