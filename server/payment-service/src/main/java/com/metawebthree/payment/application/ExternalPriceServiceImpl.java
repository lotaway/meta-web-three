package com.metawebthree.payment.application;

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

@Service
@RequiredArgsConstructor
@Slf4j
public class ExternalPriceServiceImpl {

    private final OkHttpClient httpClient;
    private final ObjectMapper objectMapper;

    public CryptoPrice fetchPrice(String symbol) {
        String[] sources = { "binance", "coinbase", "okx" };

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
                // @TODD: Get extra data
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
            default:
                throw new IllegalArgumentException("Unsupported source: " + source);
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