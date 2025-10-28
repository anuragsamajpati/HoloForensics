"""
Django REST API views for export and reporting capabilities.
Handles PDF reports, data exports, video annotations, and 3D scene exports.
"""

import json
import os
import io
from datetime import datetime
from django.http import JsonResponse, HttpResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.template.loader import render_to_string
import logging

# Import libraries for report generation
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

logger = logging.getLogger(__name__)

@login_required
@require_http_methods(["POST"])
def export_case_report(request, case_id):
    """
    Generate and export comprehensive case report in PDF format.
    """
    try:
        data = json.loads(request.body) if request.body else {}
        export_format = data.get('format', 'pdf')
        include_sections = data.get('sections', ['summary', 'analysis', 'timeline', 'evidence'])
        
        if export_format == 'pdf':
            return generate_pdf_report(case_id, include_sections, request.user)
        elif export_format == 'json':
            return generate_json_export(case_id, request.user)
        elif export_format == 'csv':
            return generate_csv_export(case_id, request.user)
        else:
            return JsonResponse({
                "success": False,
                "error": f"Unsupported export format: {export_format}"
            }, status=400)
            
    except Exception as e:
        logger.error(f"Error exporting case report for {case_id}: {str(e)}")
        return JsonResponse({
            "success": False,
            "error": f"Failed to export report: {str(e)}"
        }, status=500)

def generate_pdf_report(case_id, sections, user):
    """
    Generate PDF report using ReportLab.
    """
    if not REPORTLAB_AVAILABLE:
        return generate_mock_pdf_report(case_id, sections, user)
    
    try:
        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#ff4757'),
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2d3436'),
            spaceBefore=20,
            spaceAfter=10
        )
        
        # Mock case data
        case_data = get_mock_case_data(case_id)
        
        # Title
        story.append(Paragraph("HoloForensics Case Report", title_style))
        story.append(Paragraph(f"Case ID: {case_id}", styles['Heading3']))
        story.append(Spacer(1, 20))
        
        # Case Summary
        if 'summary' in sections:
            story.append(Paragraph("Case Summary", heading_style))
            summary_data = [
                ['Title:', case_data['title']],
                ['Status:', case_data['status'].title()],
                ['Priority:', case_data['priority'].title()],
                ['Created By:', case_data['created_by']],
                ['Created Date:', case_data['created_at'][:10]],
                ['Last Updated:', case_data['updated_at'][:10]],
                ['Case Type:', case_data['case_type'].replace('_', ' ').title()]
            ]
            
            summary_table = Table(summary_data, colWidths=[2*inch, 4*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8f9fa')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6'))
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 20))
        
        # Analysis Results
        if 'analysis' in sections:
            story.append(Paragraph("Analysis Results", heading_style))
            
            analysis_data = [
                ['Objects Detected:', str(case_data['analysis_results']['objects_detected'])],
                ['Trajectories Analyzed:', str(case_data['analysis_results']['trajectories_analyzed'])],
                ['Events Identified:', str(case_data['analysis_results']['events_identified'])],
                ['Anomalies Found:', str(case_data['analysis_results']['anomalies_found'])],
                ['Confidence Score:', f"{case_data['analysis_results']['confidence_score']}%"],
                ['Processing Time:', case_data['analysis_results']['processing_time']],
                ['Algorithms Used:', ', '.join(case_data['analysis_results']['algorithms_used'])]
            ]
            
            analysis_table = Table(analysis_data, colWidths=[2*inch, 4*inch])
            analysis_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f5e8')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6'))
            ]))
            story.append(analysis_table)
            story.append(Spacer(1, 20))
        
        # Timeline
        if 'timeline' in sections:
            story.append(Paragraph("Case Timeline", heading_style))
            
            timeline_data = [['Timestamp', 'Action', 'User', 'Details']]
            for entry in case_data['timeline']:
                timeline_data.append([
                    entry['timestamp'][:16].replace('T', ' '),
                    entry['action'].replace('_', ' ').title(),
                    entry['user'],
                    entry['details'][:50] + '...' if len(entry['details']) > 50 else entry['details']
                ])
            
            timeline_table = Table(timeline_data, colWidths=[1.5*inch, 1.2*inch, 1*inch, 2.3*inch])
            timeline_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff4757')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
            ]))
            story.append(timeline_table)
            story.append(Spacer(1, 20))
        
        # Evidence Files
        if 'evidence' in sections:
            story.append(Paragraph("Evidence Files", heading_style))
            
            evidence_data = [['File Name', 'Type', 'Size', 'Status', 'Uploaded By']]
            for evidence in case_data['evidence_files']:
                evidence_data.append([
                    evidence['name'],
                    evidence['type'].title(),
                    evidence['size'],
                    evidence['status'].title(),
                    evidence['uploaded_by']
                ])
            
            evidence_table = Table(evidence_data, colWidths=[2*inch, 0.8*inch, 0.8*inch, 1*inch, 1.4*inch])
            evidence_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e90ff')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
            ]))
            story.append(evidence_table)
        
        # Footer
        story.append(Spacer(1, 30))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#6c757d'),
            alignment=1
        )
        story.append(Paragraph(
            f"Generated by HoloForensics on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | User: {user.username}",
            footer_style
        ))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        # Create response
        response = HttpResponse(buffer.getvalue(), content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{case_id}_report.pdf"'
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        return generate_mock_pdf_report(case_id, sections, user)

def generate_mock_pdf_report(case_id, sections, user):
    """
    Generate a mock PDF report when ReportLab is not available.
    """
    # Create a simple text-based report
    report_content = f"""
HoloForensics Case Report
========================

Case ID: {case_id}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Generated by: {user.username}

Case Summary:
- Title: Indoor Incident Analysis
- Status: Completed
- Priority: High
- Created: 2024-01-15
- Case Type: Incident Analysis

Analysis Results:
- Objects Detected: 47
- Trajectories Analyzed: 12
- Events Identified: 8
- Anomalies Found: 3
- Confidence Score: 94.2%
- Processing Time: 2h 34m

Timeline:
- 2024-01-15 10:30: Case created
- 2024-01-15 11:15: Analysis started
- 2024-01-15 14:30: Analysis completed
- 2024-01-15 16:45: Case reviewed and completed

Evidence Files:
- camera_01_footage.mp4 (2.1 GB) - Processed
- camera_02_footage.mp4 (1.8 GB) - Processed

This report was generated by HoloForensics Advanced Analytics System.
"""
    
    response = HttpResponse(report_content, content_type='text/plain')
    response['Content-Disposition'] = f'attachment; filename="{case_id}_report.txt"'
    
    return response

def generate_json_export(case_id, user):
    """
    Generate JSON export of case data.
    """
    try:
        case_data = get_mock_case_data(case_id)
        
        # Add export metadata
        export_data = {
            "export_info": {
                "case_id": case_id,
                "exported_by": user.username,
                "exported_at": datetime.now().isoformat(),
                "format": "json",
                "version": "1.0"
            },
            "case_data": case_data
        }
        
        response = JsonResponse(export_data, json_dumps_params={'indent': 2})
        response['Content-Disposition'] = f'attachment; filename="{case_id}_data.json"'
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating JSON export: {str(e)}")
        return JsonResponse({
            "success": False,
            "error": f"Failed to generate JSON export: {str(e)}"
        }, status=500)

def generate_csv_export(case_id, user):
    """
    Generate CSV export of case analysis results.
    """
    try:
        import csv
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(['Case ID', 'Timestamp', 'Object Type', 'Confidence', 'X', 'Y', 'Z'])
        
        # Mock detection data
        detections = [
            [case_id, '2024-01-15T10:30:00Z', 'person', 0.95, -2.5, 1.0, 0.0],
            [case_id, '2024-01-15T10:30:01Z', 'person', 0.94, -2.4, 1.1, 0.0],
            [case_id, '2024-01-15T10:30:02Z', 'person', 0.96, -2.3, 1.2, 0.0],
            [case_id, '2024-01-15T10:30:03Z', 'vehicle', 0.92, -5.0, -3.0, 0.0],
            [case_id, '2024-01-15T10:30:04Z', 'person', 0.88, 3.2, -1.5, 0.0]
        ]
        
        for detection in detections:
            writer.writerow(detection)
        
        response = HttpResponse(output.getvalue(), content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{case_id}_detections.csv"'
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating CSV export: {str(e)}")
        return JsonResponse({
            "success": False,
            "error": f"Failed to generate CSV export: {str(e)}"
        }, status=500)

@login_required
@require_http_methods(["POST"])
def export_dashboard_report(request):
    """
    Export comprehensive dashboard analytics report.
    """
    try:
        data = json.loads(request.body) if request.body else {}
        export_format = data.get('format', 'pdf')
        date_range = data.get('date_range', 30)  # days
        
        if export_format == 'pdf':
            return generate_dashboard_pdf(date_range, request.user)
        elif export_format == 'json':
            return generate_dashboard_json(date_range, request.user)
        else:
            return JsonResponse({
                "success": False,
                "error": f"Unsupported export format: {export_format}"
            }, status=400)
            
    except Exception as e:
        logger.error(f"Error exporting dashboard report: {str(e)}")
        return JsonResponse({
            "success": False,
            "error": f"Failed to export dashboard report: {str(e)}"
        }, status=500)

def generate_dashboard_pdf(date_range, user):
    """
    Generate dashboard analytics PDF report.
    """
    dashboard_content = f"""
HoloForensics Analytics Dashboard Report
=======================================

Report Period: Last {date_range} days
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Generated by: {user.username}

Key Metrics:
- Total Cases: 156
- Active Cases: 12
- Completed Cases: 134
- Success Rate: 96.8%
- Average Processing Time: 4.2 hours

Case Distribution by Type:
- Incident Analysis: 45 cases
- Scene Reconstruction: 38 cases
- Behavioral Analysis: 32 cases
- Evidence Timeline: 28 cases
- Object Tracking: 13 cases

Performance Metrics:
- Objects Detected: 1,247
- Trajectories Analyzed: 89
- Anomalies Found: 23
- Average Confidence: 94.2%

System Performance:
- Average CPU Usage: 65%
- Average GPU Usage: 78%
- Average Memory Usage: 42%
- Uptime: 99.7%

This report provides insights into the HoloForensics system performance
and case processing statistics for the specified time period.
"""
    
    response = HttpResponse(dashboard_content, content_type='text/plain')
    response['Content-Disposition'] = f'attachment; filename="dashboard_report_{datetime.now().strftime("%Y%m%d")}.txt"'
    
    return response

def generate_dashboard_json(date_range, user):
    """
    Generate dashboard analytics JSON export.
    """
    dashboard_data = {
        "export_info": {
            "type": "dashboard_analytics",
            "date_range_days": date_range,
            "exported_by": user.username,
            "exported_at": datetime.now().isoformat(),
            "format": "json",
            "version": "1.0"
        },
        "metrics": {
            "total_cases": 156,
            "active_cases": 12,
            "completed_cases": 134,
            "success_rate": 96.8,
            "average_processing_time_hours": 4.2,
            "objects_detected": 1247,
            "trajectories_analyzed": 89,
            "anomalies_found": 23,
            "average_confidence": 94.2
        },
        "case_distribution": {
            "incident_analysis": 45,
            "scene_reconstruction": 38,
            "behavioral_analysis": 32,
            "evidence_timeline": 28,
            "object_tracking": 13
        },
        "performance": {
            "avg_cpu_usage": 65,
            "avg_gpu_usage": 78,
            "avg_memory_usage": 42,
            "uptime_percentage": 99.7
        }
    }
    
    response = JsonResponse(dashboard_data, json_dumps_params={'indent': 2})
    response['Content-Disposition'] = f'attachment; filename="dashboard_analytics_{datetime.now().strftime("%Y%m%d")}.json"'
    
    return response

def get_mock_case_data(case_id):
    """
    Get mock case data for report generation.
    """
    return {
        "id": case_id,
        "title": "Indoor Incident Analysis",
        "description": "Multi-camera analysis of indoor incident with 4 camera angles and object tracking",
        "status": "completed",
        "priority": "high",
        "created_by": "investigator1",
        "created_at": "2024-01-15T10:30:00Z",
        "updated_at": "2024-01-15T16:45:00Z",
        "case_type": "incident_analysis",
        "evidence_files": [
            {
                "name": "camera_01_footage.mp4",
                "type": "video",
                "size": "2.1 GB",
                "status": "processed",
                "uploaded_by": "investigator1"
            },
            {
                "name": "camera_02_footage.mp4",
                "type": "video", 
                "size": "1.8 GB",
                "status": "processed",
                "uploaded_by": "investigator1"
            }
        ],
        "analysis_results": {
            "objects_detected": 47,
            "trajectories_analyzed": 12,
            "events_identified": 8,
            "anomalies_found": 3,
            "confidence_score": 94.2,
            "processing_time": "2h 34m",
            "algorithms_used": ["YOLO", "DeepSORT", "Kalman Filter"]
        },
        "timeline": [
            {
                "timestamp": "2024-01-15T10:30:00Z",
                "action": "case_created",
                "user": "investigator1",
                "details": "Case created and initial evidence uploaded"
            },
            {
                "timestamp": "2024-01-15T11:15:00Z",
                "action": "analysis_started",
                "user": "system",
                "details": "Automated analysis pipeline initiated"
            },
            {
                "timestamp": "2024-01-15T14:30:00Z",
                "action": "analysis_completed",
                "user": "system",
                "details": "Analysis completed with 94.2% confidence"
            },
            {
                "timestamp": "2024-01-15T16:45:00Z",
                "action": "case_reviewed",
                "user": "analyst2",
                "details": "Case reviewed and marked as completed"
            }
        ]
    }
