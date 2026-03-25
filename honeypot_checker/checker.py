# honeypot_checker/checker.py
import requests
import streamlit as st

# -------------------- EVM (Ethereum, BSC, Base) --------------------
def check_honeypot(token_address, chain_id=None):
    """Check if an EVM token is a honeypot using honeypot.is API."""
    try:
        url = "https://api.honeypot.is/v2/IsHoneypot"
        params = {"address": token_address}
        if chain_id:
            params["chainID"] = chain_id
        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            return {"error": f"API error: {response.status_code}"}
        data = response.json()
        return {
            "success": True,
            "type": "evm",
            "token_name": data.get("token", {}).get("name", "N/A"),
            "token_symbol": data.get("token", {}).get("symbol", "N/A"),
            "risk": data.get("summary", {}).get("risk", "unknown"),
            "risk_level": data.get("summary", {}).get("riskLevel", 0),
            "is_honeypot": data.get("honeypotResult", {}).get("isHoneypot", False),
            "honeypot_reason": data.get("honeypotResult", {}).get("honeypotReason", None),
            "buy_tax": data.get("simulationResult", {}).get("buyTax", 0),
            "sell_tax": data.get("simulationResult", {}).get("sellTax", 0),
            "transfer_tax": data.get("simulationResult", {}).get("transferTax", 0),
            "open_source": data.get("contractCode", {}).get("openSource", False),
            "holders": data.get("token", {}).get("totalHolders", "N/A"),
            "liquidity_usd": data.get("pair", {}).get("liquidity", 0),
            "chain": data.get("chain", {}).get("name", "Unknown"),
            "flags": data.get("flags", [])
        }
    except Exception as e:
        return {"error": str(e)}


# -------------------- Solana (RugCheck) --------------------
def check_solana_token(token_address):
    """Check Solana token using RugCheck.xyz API."""
    try:
        url = f"https://api.rugcheck.xyz/v1/tokens/{token_address}/report"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return {"error": f"RugCheck API error: {response.status_code}"}
        data = response.json()
        return {
            "success": True,
            "type": "solana",
            "chain": "Solana",
            "token_name": data.get("tokenMeta", {}).get("name", "N/A"),
            "token_symbol": data.get("tokenMeta", {}).get("symbol", "N/A"),
            "risk_score": data.get("riskScore", 0),
            "risk_level": data.get("riskLevel", "unknown"),
            "is_mint_authority_revoked": data.get("mintAuthorityRevoked", False),
            "is_freeze_authority_revoked": data.get("freezeAuthorityRevoked", False),
            "total_supply": data.get("supply", 0),
            "holders": data.get("holderCount", 0),
            "liquidity_usd": data.get("liquidity", 0),
            "flags": data.get("flags", [])
        }
    except Exception as e:
        return {"error": str(e)}


# -------------------- pump.fun (DexScreener fallback) --------------------
def check_pumpfun_token(mint_address):
    """Check pump.fun token via DexScreener and provide risk analysis."""
    try:
        url = f"https://api.dexscreener.com/latest/dex/search?q={mint_address}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("pairs"):
                pair = data["pairs"][0]
                return {
                    "success": True,
                    "type": "pumpfun",
                    "chain": "Solana (pump.fun)",
                    "token_name": pair.get("baseToken", {}).get("name", "Unknown"),
                    "token_symbol": pair.get("baseToken", {}).get("symbol", "Unknown"),
                    "price_usd": float(pair.get("priceUsd", 0)),
                    "liquidity_usd": float(pair.get("liquidity", {}).get("usd", 0)),
                    "volume_24h": float(pair.get("volume", {}).get("h24", 0)),
                    "change_1h": float(pair.get("priceChange", {}).get("h1", 0)),
                    "holders": pair.get("holders", {}).get("count", 0),
                    "flags": ["High risk due to pump.fun nature"] if pair.get("fdv", 0) < 50000 else []
                }
    except Exception:
        pass
    return {
        "error": "Could not fetch data. This token may be very new or not on DexScreener.",
        "risk_level": "high",
        "message": "pump.fun tokens are extremely speculative. Always DYOR."
    }


# -------------------- Display Functions --------------------
def get_risk_color(risk_level):
    if risk_level >= 90:
        return "🔴"
    elif risk_level >= 60:
        return "🟠"
    elif risk_level >= 20:
        return "🟡"
    else:
        return "🟢"


def display_honeypot_result(result):
    """Universal display function for EVM, Solana, and pump.fun results."""
    if "error" in result:
        st.error(f"❌ {result['error']}")
        return

    token_type = result.get("type", "evm")

    # ----- Risk / Honeypot status -----
    if token_type == "evm":
        risk_icon = get_risk_color(result.get("risk_level", 0))
        if result.get("is_honeypot"):
            st.error(f"{risk_icon} **HONEYPOT DETECTED!** ⚠️")
            st.markdown(f"**Risk Level:** {result['risk_level']}/100 - {result['risk'].upper()}")
            if result.get("honeypot_reason"):
                st.warning(f"**Reason:** {result['honeypot_reason']}")
        else:
            if result['risk_level'] == 0:
                st.success(f"{risk_icon} **SAFE - Whitelisted Token** ✅")
            elif result['risk_level'] < 20:
                st.success(f"{risk_icon} **LOW RISK** ✅")
            elif result['risk_level'] < 60:
                st.warning(f"{risk_icon} **MEDIUM RISK** ⚠️")
            elif result['risk_level'] < 90:
                st.warning(f"{risk_icon} **HIGH RISK** ⚠️")
            else:
                st.error(f"{risk_icon} **VERY HIGH RISK** 🔴")
    elif token_type == "solana":
        risk_score = result.get("risk_score", 0)
        risk_level = result.get("risk_level", "unknown")
        if risk_level == "low":
            st.success(f"🟢 **LOW RISK** (Score: {risk_score})")
        elif risk_level == "medium":
            st.warning(f"🟡 **MEDIUM RISK** (Score: {risk_score})")
        else:
            st.error(f"🔴 **HIGH RISK** (Score: {risk_score})")
    else:  # pump.fun
        st.warning("⚠️ **pump.fun Token – Extremely Speculative**")
        st.info("No automated honeypot check available. Use DexScreener data and DYOR.")

    # ----- Token Information -----
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Token:** {result.get('token_name', 'N/A')} ({result.get('token_symbol', 'N/A')})")
        st.markdown(f"**Chain:** {result.get('chain', 'Unknown')}")
        if token_type == "evm":
            st.markdown(f"**Holders:** {result.get('holders', 'N/A')}")
        elif token_type in ("solana", "pumpfun"):
            st.markdown(f"**Holders:** {result.get('holders', 'N/A')}")
    with col2:
        if token_type == "evm":
            st.markdown(f"**Open Source:** {'✅ Yes' if result.get('open_source') else '❌ No'}")
            st.markdown(f"**Liquidity:** ${result.get('liquidity_usd', 0):,.2f}" if result.get('liquidity_usd') else "**Liquidity:** N/A")
        elif token_type == "solana":
            st.markdown(f"**Liquidity:** ${result.get('liquidity_usd', 0):,.2f}" if result.get('liquidity_usd') else "**Liquidity:** N/A")
            if result.get('is_mint_authority_revoked') is not None:
                st.markdown(f"**Mint Authority:** {'Revoked ✅' if result['is_mint_authority_revoked'] else 'Still Active ❌'}")
                st.markdown(f"**Freeze Authority:** {'Revoked ✅' if result['is_freeze_authority_revoked'] else 'Still Active ❌'}")
        elif token_type == "pumpfun":
            st.markdown(f"**Liquidity:** ${result.get('liquidity_usd', 0):,.2f}" if result.get('liquidity_usd') else "**Liquidity:** N/A")
            st.markdown(f"**24h Volume:** ${result.get('volume_24h', 0):,.2f}" if result.get('volume_24h') else "**24h Volume:** N/A")
            st.markdown(f"**1h Change:** {result.get('change_1h', 0):.2f}%")

    # ----- Taxes (EVM only) -----
    if token_type == "evm":
        st.divider()
        st.subheader("💰 Transaction Taxes")
        tax_col1, tax_col2, tax_col3 = st.columns(3)
        with tax_col1:
            st.metric("Buy Tax", f"{result['buy_tax']}%")
        with tax_col2:
            st.metric("Sell Tax", f"{result['sell_tax']}%")
        with tax_col3:
            st.metric("Transfer Tax", f"{result['transfer_tax']}%")

    # ----- Flags / Warnings -----
    if result.get("flags"):
        st.divider()
        st.subheader("⚠️ Warnings / Flags")
        for flag in result["flags"]:
            if isinstance(flag, dict):
                st.warning(f"**{flag.get('flag', 'Warning')}**: {flag.get('description', '')}")
            else:
                st.warning(f"⚠️ {flag}")