# New positions function with --json flag

def positions(
    expiring: bool = typer.Option(False, "--expiring", "-e", help="Sort by soonest expiry"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Show current positions"""
    try:
        data = api("GET", "portfolio/positions")
    except ApiError as e:
        handle_api_error(e)
    ps = data.get("market_positions", [])
    ps = [p for p in ps if p.get("position", 0) != 0]
    if not ps:
        console.print("[dim]No open positions[/dim]")
        raise typer.Exit()

    with console.status("[dim]Loading position details...[/dim]"):
        ps = _enrich_positions(ps)

    if expiring:
        ps.sort(key=lambda p: p.get("expiration_time", "") or "9999")

    if json_output:
        import json
        # Convert to serializable format
        output = []
        for p in ps:
            title = _normalize_title(p.get("title", "") or "")
            qty = abs(p.get("position", 0))
            total_cost = float(p.get("total_traded_dollars", 0) or 0)
            cost_per_share = (total_cost / qty) if qty > 0 else 0
            
            if p.get("position", 0) > 0:
                val_per_share = float(p.get("yes_bid_dollars", 0) or 0)
            else:
                val_per_share = float(p.get("no_bid_dollars", 0) or 0)
            
            current_value = _position_current_value(p)
            
            output.append({
                "ticker": p.get("ticker", ""),
                "title": title,
                "side": "YES" if p.get("position", 0) > 0 else "NO",
                "quantity": qty,
                "cost_per_share": round(cost_per_share, 2),
                "current_value_per_share": round(val_per_share, 2),
                "total_value": round(current_value, 2),
                "return_pct": round((val_per_share - cost_per_share) / cost_per_share * 100 if cost_per_share > 0 else 0, 1),
                "expiration": p.get("expiration_time", "")[:10] if p.get("expiration_time") else ""
            })
        
        console.print(json.dumps(output, indent=2))
        raise typer.Exit()

    # Original table output
    t = Table(title="Current Positions", box=box.ROUNDED)
    t.add_column("Title", max_width=40)
    t.add_column("Side", max_width=25)
    t.add_column("Qty", justify="right")
    t.add_column("Cost/Sh", justify="right", style="yellow")
    t.add_column("Val/Sh", justify="right")
    t.add_column("Return", justify="right")
    t.add_column("Value", justify="right")
    t.add_column("Expires", style="dim")
    t.add_column("Ticker", style="cyan", overflow="fold")

    for p in ps:
        title = _normalize_title(p.get("title", "") or "")
        side_str = _position_outcome_label(p)
        qty = abs(p.get("position", 0))
        total_cost = float(p.get("total_traded_dollars", 0) or 0)
        cost_per_share = (total_cost / qty) if qty > 0 else 0

        if p.get("position", 0) > 0:
            val_per_share = float(p.get("yes_bid_dollars", 0) or 0)
        else:
            val_per_share = float(p.get("no_bid_dollars", 0) or 0)

        current_value = _position_current_value(p)

        t.add_row(
            title,
            side_str,
            str(qty),
            f"${cost_per_share:.2f}" if cost_per_share > 0 else "—",
            f"${val_per_share:.2f}" if val_per_share > 0 else "—",
            _position_return_pct(p),
            fmt_dollars(current_value),
            fmt_expiry(p.get("expiration_time", "")),
            p.get("ticker", ""),
        )
    console.print(t)
