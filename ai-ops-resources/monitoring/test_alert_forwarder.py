#!/usr/bin/env python3
"""
Test script to send sample alerts to the alert-forwarder.
"""

import requests
import json
from datetime import datetime, timezone

def send_test_alert():
    """Send a test alert to the alert-forwarder."""
    # Sample Prometheus alert payload - Database connection issue
    alert_payload = {
        "alerts": [
            {
                "status": "firing",
                "labels": {
                    "alertname": "DatabaseConnectionFailure",
                    "severity": "critical",
                    "instance": "db-server-01:5432",
                    "job": "postgresql",
                    "database": "users_db",
                    "environment": "production"
                },
                "annotations": {
                    "summary": "Database connection failure detected",
                    "description": "Unable to establish connection to PostgreSQL database. Connection timeout after 30 seconds. Multiple connection attempts failed.",
                    "runbook_url": "https://company.com/runbooks/database-connection-failure"
                },
                "startsAt": datetime.now(timezone.utc).isoformat(),
                "endsAt": "0001-01-01T00:00:00Z",
                "generatorURL": "http://localhost:9090/graph?g0.expr=postgresql_up%3D%3D0",
                "fingerprint": "db1234567890abcdef"
            }
        ]
    }
    
    try:
        # Send POST request to alert-forwarder
        response = requests.post(
            "http://localhost:8000/alerts",
            json=alert_payload,
            headers={"Content-Type": "application/json"}
        )
        
        response.raise_for_status()
        result = response.json()
        
        print(f"✅ Successfully sent alerts: {result}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error sending alerts: {e}")

def send_memory_alert():
    """Send a memory exhaustion alert."""
    alert_payload = {
        "alerts": [
            {
                "status": "firing",
                "labels": {
                    "alertname": "HighMemoryUsage",
                    "severity": "warning",
                    "instance": "web-server-03:9100",
                    "job": "node_exporter",
                    "service": "nginx",
                    "environment": "production"
                },
                "annotations": {
                    "summary": "Memory usage is critically high",
                    "description": "Memory usage has exceeded 90% for the last 10 minutes on web-server-03. Current usage: 15.2GB/16GB",
                    "impact": "Service performance degradation expected"
                },
                "startsAt": datetime.now(timezone.utc).isoformat(),
                "endsAt": "0001-01-01T00:00:00Z",
                "generatorURL": "http://localhost:9090/graph?g0.expr=memory_usage_percent%3E90",
                "fingerprint": "mem9876543210fedcba"
            }
        ]
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/alerts",
            json=alert_payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        print(f"✅ Successfully sent memory alert: {result}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error sending memory alert: {e}")

def send_network_alert():
    """Send a network connectivity alert."""
    alert_payload = {
        "alerts": [
            {
                "status": "firing",
                "labels": {
                    "alertname": "NetworkLatencyHigh",
                    "severity": "major",
                    "instance": "api-gateway-01:8080",
                    "job": "api-gateway",
                    "region": "us-east-1",
                    "environment": "production"
                },
                "annotations": {
                    "summary": "High network latency detected",
                    "description": "Average response time has increased to 2.5 seconds (normal: <500ms). External API calls timing out.",
                    "troubleshooting": "Check upstream service health and network connectivity"
                },
                "startsAt": datetime.now(timezone.utc).isoformat(),
                "endsAt": "0001-01-01T00:00:00Z",
                "generatorURL": "http://localhost:9090/graph?g0.expr=http_request_duration_seconds%3E2",
                "fingerprint": "net5432109876abcdef"
            }
        ]
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/alerts",
            json=alert_payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        print(f"✅ Successfully sent network alert: {result}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error sending network alert: {e}")

def send_application_alert():
    """Send an application error alert."""
    alert_payload = {
        "alerts": [
            {
                "status": "firing",
                "labels": {
                    "alertname": "ApplicationErrorRate",
                    "severity": "critical",
                    "instance": "app-server-02:3000",
                    "job": "nodejs-app",
                    "service": "user-service",
                    "environment": "production"
                },
                "annotations": {
                    "summary": "High application error rate detected",
                    "description": "Error rate has spiked to 15% (threshold: 5%). 500 Internal Server Errors increasing. Last error: 'Database connection pool exhausted'",
                    "dashboard": "https://grafana.company.com/d/app-errors",
                    "logs": "https://kibana.company.com/app/logs?filter=service:user-service"
                },
                "startsAt": datetime.now(timezone.utc).isoformat(),
                "endsAt": "0001-01-01T00:00:00Z",
                "generatorURL": "http://localhost:9090/graph?g0.expr=error_rate%3E0.05",
                "fingerprint": "app1357924680bdfeca"
            }
        ]
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/alerts",
            json=alert_payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        print(f"✅ Successfully sent application alert: {result}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error sending application alert: {e}")

def check_health():
    """Check the health of the alert-forwarder."""
    try:
        response = requests.get("http://localhost:8000/health")
        response.raise_for_status()
        result = response.json()
        print(f"🏥 Health check: {result}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")

if __name__ == "__main__":
    # send_test_alert()
    # send_memory_alert()
    # send_network_alert()
    send_application_alert()