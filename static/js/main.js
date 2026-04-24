/**
 * TruthLens AI — Main JavaScript
 * Minimal JS — no dark mode needed.
 */

document.addEventListener('DOMContentLoaded', function() {
    // Smooth page entrance animation
    document.body.style.opacity = '0';
    document.body.style.transition = 'opacity 0.4s ease';
    requestAnimationFrame(() => {
        document.body.style.opacity = '1';
    });
});
