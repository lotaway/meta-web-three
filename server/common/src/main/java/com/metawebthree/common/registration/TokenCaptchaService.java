package com.metawebthree.common.registration;

import com.metawebthree.common.services.DistributedCacheService;

import javax.imageio.ImageIO;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.security.SecureRandom;
import java.util.Base64;
import java.util.concurrent.TimeUnit;

public class TokenCaptchaService {

    private static final String CHAR_POOL = "ABCDEFGHJKMNPQRSTUVWXYZ23456789";
    private static final int CODE_LENGTH = 4;
    private static final int IMAGE_WIDTH = 150;
    private static final int IMAGE_HEIGHT = 52;
    private static final int NOISE_LINES = 6;
    private static final int NOISE_DOTS = 50;
    private static final SecureRandom RANDOM = new SecureRandom();

    private final DistributedCacheService cacheService;
    private final String cacheName;
    private final long ttl;
    private final TimeUnit ttlUnit;

    public TokenCaptchaService(DistributedCacheService cacheService, String cacheName,
                               long ttl, TimeUnit ttlUnit) {
        this.cacheService = cacheService;
        this.cacheName = cacheName;
        this.ttl = ttl;
        this.ttlUnit = ttlUnit;
    }

    public CaptchaChallenge generate() {
        String code = randomCode();
        String token = java.util.UUID.randomUUID().toString();
        cacheService.put(cacheName, token, code, ttl, ttlUnit);
        return new CaptchaChallenge(token, renderImage(code));
    }

    public boolean verify(String token, String answer) {
        if (token == null || answer == null) {
            return false;
        }
        String cached = cacheService.get(cacheName, token);
        cacheService.evict(cacheName, token);
        if (cached == null) {
            return false;
        }
        return cached.equals(answer.trim().toUpperCase());
    }

    private String randomCode() {
        StringBuilder code = new StringBuilder(CODE_LENGTH);
        for (int i = 0; i < CODE_LENGTH; i++) {
            code.append(CHAR_POOL.charAt(RANDOM.nextInt(CHAR_POOL.length())));
        }
        return code.toString();
    }

    private String renderImage(String code) {
        BufferedImage image = new BufferedImage(IMAGE_WIDTH, IMAGE_HEIGHT, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = image.createGraphics();
        try {
            drawBackground(graphics);
            drawNoiseLines(graphics);
            drawCharacters(graphics, code);
            drawNoiseDots(graphics);
        } finally {
            graphics.dispose();
        }
        return toBase64(image);
    }

    private void drawBackground(Graphics2D graphics) {
        graphics.setColor(new Color(244, 245, 247));
        graphics.fillRect(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);
    }

    private void drawNoiseLines(Graphics2D graphics) {
        graphics.setStroke(new BasicStroke(1.4f));
        for (int i = 0; i < NOISE_LINES; i++) {
            graphics.setColor(randomColor());
            graphics.drawLine(RANDOM.nextInt(IMAGE_WIDTH), RANDOM.nextInt(IMAGE_HEIGHT),
                    RANDOM.nextInt(IMAGE_WIDTH), RANDOM.nextInt(IMAGE_HEIGHT));
        }
    }

    private void drawCharacters(Graphics2D graphics, String code) {
        int x = 14;
        int baseline = IMAGE_HEIGHT / 2 + 10;
        for (char character : code.toCharArray()) {
            graphics.setFont(new Font("Dialog", Font.BOLD, 28 + RANDOM.nextInt(6)));
            graphics.setColor(randomColor());
            double rotation = (RANDOM.nextDouble() - 0.5) * 0.6;
            graphics.rotate(rotation, x, baseline);
            graphics.drawString(String.valueOf(character), x, baseline);
            graphics.rotate(-rotation, x, baseline);
            x += 30;
        }
    }

    private void drawNoiseDots(Graphics2D graphics) {
        for (int i = 0; i < NOISE_DOTS; i++) {
            graphics.setColor(randomColor());
            graphics.fillOval(RANDOM.nextInt(IMAGE_WIDTH), RANDOM.nextInt(IMAGE_HEIGHT), 2, 2);
        }
    }

    private Color randomColor() {
        return new Color(40 + RANDOM.nextInt(120), 40 + RANDOM.nextInt(120), 40 + RANDOM.nextInt(120));
    }

    private String toBase64(BufferedImage image) {
        try {
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            ImageIO.write(image, "png", output);
            return "data:image/png;base64," + Base64.getEncoder().encodeToString(output.toByteArray());
        } catch (IOException e) {
            throw new IllegalStateException("Failed to render captcha image", e);
        }
    }

    public record CaptchaChallenge(String token, String image) {
    }
}