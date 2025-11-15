def retrieve_vix_from_market_data(market_data):
    """Retrieve the VIX value from market data."""
    vix_display = 'N/A'
    vix_trend = None
    if market_data:
        for item in market_data:
            content = item.get('content', {})
            if 'VIX' in content.get('indicator', '').upper():
                vix_display = content.get('value', 'N/A')
                trend = content.get('trend', 'N/A')
                if trend == 'Growing':
                    vix_trend = "↑"
                elif trend == 'Decreasing':
                    vix_trend = "↓"
                try:
                    vix_display = f"{float(vix_display):.2f}"
                except (ValueError, TypeError):
                    pass
                break
    return vix_display, vix_trend