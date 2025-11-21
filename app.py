# INSTALL REQUIRED PACKAGES
!pip install -q gradio PyPDF2 pdfplumber pandas numpy plotly groq google-generativeai openai

import os
import re
import gradio as gr
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
import pdfplumber
import json
import tempfile
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from datetime import datetime

# ============================================================================
# FINANCIAL REPORT PROCESSOR - SPECIFICALLY FOR LATVIAN REPORTS
# ============================================================================

class LatvianFinancialReportProcessor:
    """Specialized processor for Latvian financial reports with correct number extraction"""
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using multiple methods"""
        text = ""
        
        # Try pdfplumber first (better for tables)
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                return text
        except Exception as e:
            print(f"pdfplumber failed: {e}")
        
        # Fallback to PyPDF2
        try:
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            print(f"PyPDF2 failed: {e}")
            return ""
    
    def clean_latvian_number(self, number_str: str) -> float:
        """Convert Latvian number format (spaces as thousands separators) to float"""
        if not number_str or number_str.strip() == '':
            return 0.0
        
        # Remove spaces (thousands separators), keep minus sign and digits
        clean_str = re.sub(r'[^\d\-]', '', str(number_str).replace(' ', '').replace('\xa0', ''))
        
        try:
            return float(clean_str) if clean_str else 0.0
        except ValueError:
            return 0.0

    def extract_financial_data(self, text: str) -> Dict[str, float]:
        """Extract financial data from Latvian report text using line codes"""
        results = {}
        
        # Define line code mappings for Latvian financial reports
        line_code_mapping = {
            # Balance Sheet - Assets
            '640': 'total_assets',
            '630': 'current_assets', 
            '470': 'receivables',
            '480': 'related_party_receivables',
            '620': 'cash',
            '550': 'total_receivables',
            
            # Balance Sheet - Liabilities & Equity
            '800': 'equity',
            '660': 'share_capital',
            '780': 'retained_earnings',
            '790': 'net_income',
            '1180': 'current_liabilities',
            '1080': 'suppliers_debt',
            '1120': 'taxes_liability',
            '1130': 'other_creditors',
            
            # Income Statement
            '10': 'revenue',
            '40': 'cogs',
            '50': 'gross_profit',
            '70': 'admin_expenses',
            '80': 'other_income',
            '240': 'profit_before_tax',
            '250': 'income_tax',
            '260': 'profit_after_tax',
            '290': 'net_income_pza'
        }
        
        # Extract all line codes with their values
        lines = text.split('\n')
        
        for line in lines:
            # Look for pattern: | line_code | description | current_year | previous_year |
            pattern = r'\|\s*(\d+)\s*\|\s*[^|]*\s*\|\s*([-\d\s]+)\s*\|\s*[-\d\s]*\s*\|'
            matches = re.findall(pattern, line)
            
            for match in matches:
                line_code, value_str = match
                if line_code in line_code_mapping:
                    field_name = line_code_mapping[line_code]
                    value = self.clean_latvian_number(value_str)
                    results[field_name] = value
        
        # Alternative extraction for key totals
        if 'total_assets' not in results:
            # Look for BILANCE pattern
            bilance_pattern = r'BILANCE.*?\|.*?\|.*?\|\s*([-\d\s]+)'
            bilance_match = re.search(bilance_pattern, text)
            if bilance_match:
                results['total_assets'] = self.clean_latvian_number(bilance_match.group(1))
        
        if 'equity' not in results:
            # Look for equity pattern
            equity_pattern = r'Pašu kapitāls kopā.*?\|.*?\|.*?\|\s*([-\d\s]+)'
            equity_match = re.search(equity_pattern, text)
            if equity_match:
                results['equity'] = self.clean_latvian_number(equity_match.group(1))
        
        if 'current_liabilities' not in results:
            # Look for current liabilities pattern
            cl_pattern = r'Īstermiņa kreditori kopā.*?\|.*?\|.*?\|\s*([-\d\s]+)'
            cl_match = re.search(cl_pattern, text)
            if cl_match:
                results['current_liabilities'] = self.clean_latvian_number(cl_match.group(1))
        
        # Calculate derived values
        if 'total_assets' in results and 'equity' in results:
            results['total_liabilities'] = results['total_assets'] - results['equity']
        
        # Ensure we have net income
        if 'net_income' not in results and 'net_income_pza' in results:
            results['net_income'] = results['net_income_pza']
        elif 'net_income' not in results and 'profit_after_tax' in results:
            results['net_income'] = results['profit_after_tax']
        
        print("=== EXTRACTED FINANCIAL DATA ===")
        for key, value in results.items():
            print(f"{key}: {value:,.2f}")
        
        return results

    def extract_company_info(self, text: str) -> Dict[str, str]:
        """Extract company information from report"""
        info = {}
        
        # Extract registration number
        reg_pattern = r"Reģistrācijas numurs\s*([\d\w]+)"
        reg_match = re.search(reg_pattern, text)
        if reg_match:
            info['registration_number'] = reg_match.group(1).strip()
        
        # Extract company name
        name_pattern = r"Nosaukums\s*([^\n]+)"
        name_match = re.search(name_pattern, text)
        if name_match:
            info['company_name'] = name_match.group(1).strip()
        
        # Extract period
        period_pattern = r"Periods no\s*(\d{2}\.\d{2}\.\d{4})\s*līdz\s*(\d{2}\.\d{2}\.\d{4})"
        period_match = re.search(period_pattern, text)
        if period_match:
            info['period'] = f"{period_match.group(1)} to {period_match.group(2)}"
        
        return info

# ============================================================================
# FINANCIAL RATIO CALCULATOR
# ============================================================================

class FinancialRatioCalculator:
    """Calculate comprehensive financial ratios"""
    
    def calculate_all_ratios(self, data: Dict[str, float]) -> Dict[str, float]:
        """Calculate all possible financial ratios"""
        ratios = {}
        
        # Extract values with defaults
        ta = data.get('total_assets', 0)
        ca = data.get('current_assets', 0)
        cl = data.get('current_liabilities', 0)
        tl = data.get('total_liabilities', 0)
        eq = data.get('equity', 0)
        ni = data.get('net_income', 0)
        rev = data.get('revenue', 0)
        cash = data.get('cash', 0)
        receivables = data.get('receivables', 0)
        cogs = data.get('cogs', 0)
        gross_profit = data.get('gross_profit', 0)
        ebit = data.get('profit_before_tax', 0)
        
        # LIQUIDITY RATIOS
        if cl > 0:
            ratios['current_ratio'] = round(ca / cl, 2)
            ratios['quick_ratio'] = round((cash + receivables) / cl, 2)
            ratios['cash_ratio'] = round(cash / cl, 2)
        else:
            ratios['current_ratio'] = float('inf')
            ratios['quick_ratio'] = float('inf')
            ratios['cash_ratio'] = float('inf')
        
        # LEVERAGE RATIOS
        if eq > 0:
            ratios['debt_to_equity'] = round(tl / eq, 2)
            ratios['debt_to_assets'] = round(tl / ta, 2) if ta > 0 else 0
            ratios['equity_multiplier'] = round(ta / eq, 2) if eq > 0 else 0
        else:
            ratios['debt_to_equity'] = float('inf')
            ratios['debt_to_assets'] = 1.0 if ta > 0 else 0
            ratios['equity_multiplier'] = float('inf')
        
        ratios['debt_ratio'] = round(tl / ta, 2) if ta > 0 else 0
        ratios['equity_ratio'] = round(eq / ta, 2) if ta > 0 else 0
        
        # PROFITABILITY RATIOS
        if ta > 0:
            ratios['return_on_assets'] = round((ni / ta) * 100, 2)
        if eq > 0:
            ratios['return_on_equity'] = round((ni / eq) * 100, 2)
        if rev > 0:
            ratios['net_profit_margin'] = round((ni / rev) * 100, 2)
            ratios['gross_profit_margin'] = round((gross_profit / rev) * 100, 2) if gross_profit else 0
            ratios['operating_margin'] = round((ebit / rev) * 100, 2) if ebit else 0
        
        # EFFICIENCY RATIOS (if we have average values, use simple calculations)
        if receivables > 0 and rev > 0:
            ratios['receivables_turnover'] = round(rev / receivables, 2)
        
        # COVERAGE RATIOS
        interest_expense = data.get('interest_expense', 1)  # Default to 1 to avoid division by zero
        if interest_expense > 0 and ebit:
            ratios['interest_coverage'] = round(ebit / interest_expense, 2)
        
        # ADDITIONAL RATIOS
        if ca > 0 and cl > 0:
            ratios['working_capital'] = ca - cl
            ratios['working_capital_ratio'] = round((ca - cl) / ta, 2) if ta > 0 else 0
        
        return ratios

# ============================================================================
# AI ANALYSIS WITH MULTIPLE API SUPPORT
# ============================================================================

class FinancialAnalysisAI:
    """AI analysis with support for multiple APIs"""
    
    def __init__(self, api_key: str = None, api_provider: str = "groq"):
        self.api_key = api_key
        self.api_provider = api_provider
    
    def analyze_financials(self, company_info: Dict, financial_data: Dict, ratios: Dict, lang: str = "en") -> str:
        """Generate AI analysis of financial data"""
        
        if not self.api_key:
            return self._generate_fallback_analysis(company_info, financial_data, ratios, lang)
        
        prompt = self._create_analysis_prompt(company_info, financial_data, ratios, lang)
        
        try:
            if self.api_provider == "groq":
                return self._call_groq_api(prompt)
            elif self.api_provider == "openai":
                return self._call_openai_api(prompt)
            elif self.api_provider == "gemini":
                return self._call_gemini_api(prompt)
            else:
                return self._generate_fallback_analysis(company_info, financial_data, ratios, lang)
        except Exception as e:
            print(f"API call failed: {e}")
            return self._generate_fallback_analysis(company_info, financial_data, ratios, lang)
    
    def _create_analysis_prompt(self, company_info: Dict, financial_data: Dict, ratios: Dict, lang: str) -> str:
        """Create analysis prompt for AI"""
        
        company_name = company_info.get('company_name', 'Unknown Company')
        period = company_info.get('period', 'Unknown Period')
        
        financial_summary = "FINANCIAL DATA:\n"
        for key, value in financial_data.items():
            financial_summary += f"- {key}: €{value:,.2f}\n"
        
        ratios_summary = "FINANCIAL RATIOS:\n"
        for key, value in ratios.items():
            ratios_summary += f"- {key}: {value}\n"
        
        if lang == 'lv':
            prompt = f"""
            Analizējiet šo uzņēmuma finanšu pārskatu un sniedziet detalizētu finanšu analīzi latviešu valodā.
            
            Uzņēmums: {company_name}
            Periods: {period}
            
            {financial_summary}
            
            {ratios_summary}
            
            Lūdzu, sniedziet:
            1. Finanšu stāvokļa novērtējumu
            2. Likviditātes analīzi
            3. Parādu sloga novērtējumu  
            4. Rentabilitātes analīzi
            5. Stiprās un vājās puses
            6. Ieteikumus
            """
        else:
            prompt = f"""
            Analyze this company financial report and provide a detailed financial analysis in English.
            
            Company: {company_name}
            Period: {period}
            
            {financial_summary}
            
            {ratios_summary}
            
            Please provide:
            1. Financial health assessment
            2. Liquidity analysis
            3. Debt burden evaluation
            4. Profitability analysis
            5. Strengths and weaknesses
            6. Recommendations
            """
        
        return prompt
    
    def _call_groq_api(self, prompt: str) -> str:
        """Call Groq API"""
        from groq import Groq
        
        client = Groq(api_key=self.api_key)
        
        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API"""
        from openai import OpenAI
        
        client = OpenAI(api_key=self.api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    def _call_gemini_api(self, prompt: str) -> str:
        """Call Google Gemini API"""
        import google.generativeai as genai
        
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        response = model.generate_content(prompt)
        return response.text
    
    def _generate_fallback_analysis(self, company_info: Dict, financial_data: Dict, ratios: Dict, lang: str) -> str:
        """Generate fallback analysis when no API available"""
        
        company_name = company_info.get('company_name', 'Unknown Company')
        
        if lang == 'lv':
            analysis = f"## Finanšu Analīze: {company_name}\n\n"
            
            # Liquidity analysis
            cr = ratios.get('current_ratio', 0)
            if cr > 2:
                analysis += "✅ **LIKVIDITĀTE:** Ļoti laba (pašreizējais rādītājs > 2)\n"
            elif cr > 1:
                analysis += "⚠️ **LIKVIDITĀTE:** Pietiekama (pašreizējais rādītājs > 1)\n"
            else:
                analysis += "🔴 **LIKVIDITĀTE:** Zema (pašreizējais rādītājs < 1)\n"
            
            # Debt analysis
            de = ratios.get('debt_to_equity', 0)
            if de < 1:
                analysis += "✅ **PARĀDS:** Konservatīvs (parāds/kapitāls < 1)\n"
            elif de < 2:
                analysis += "⚠️ **PARĀDS:** Mērens (parāds/kapitāls 1-2)\n"
            else:
                analysis += "🔴 **PARĀDS:** Augsts (parāds/kapitāls > 2)\n"
            
            # Profitability analysis
            roe = ratios.get('return_on_equity', 0)
            if roe > 15:
                analysis += "✅ **RENTABILITĀTE:** Izcila (KP > 15%)\n"
            elif roe > 0:
                analysis += "⚠️ **RENTABILITĀTE:** Pozitīva (KP > 0%)\n"
            else:
                analysis += "🔴 **RENTABILITĀTE:** Zaudējumi (KP < 0%)\n"
            
            analysis += f"\n**Aktīvi kopā:** €{financial_data.get('total_assets', 0):,.2f}"
            analysis += f"\n**Pašu kapitāls:** €{financial_data.get('equity', 0):,.2f}"
            analysis += f"\n**Neto ienākumi:** €{financial_data.get('net_income', 0):,.2f}"
            
        else:
            analysis = f"## Financial Analysis: {company_name}\n\n"
            
            # Liquidity analysis
            cr = ratios.get('current_ratio', 0)
            if cr > 2:
                analysis += "✅ **LIQUIDITY:** Very strong (current ratio > 2)\n"
            elif cr > 1:
                analysis += "⚠️ **LIQUIDITY:** Adequate (current ratio > 1)\n"
            else:
                analysis += "🔴 **LIQUIDITY:** Weak (current ratio < 1)\n"
            
            # Debt analysis
            de = ratios.get('debt_to_equity', 0)
            if de < 1:
                analysis += "✅ **DEBT:** Conservative (debt/equity < 1)\n"
            elif de < 2:
                analysis += "⚠️ **DEBT:** Moderate (debt/equity 1-2)\n"
            else:
                analysis += "🔴 **DEBT:** High (debt/equity > 2)\n"
            
            # Profitability analysis
            roe = ratios.get('return_on_equity', 0)
            if roe > 15:
                analysis += "✅ **PROFITABILITY:** Excellent (ROE > 15%)\n"
            elif roe > 0:
                analysis += "⚠️ **PROFITABILITY:** Positive (ROE > 0%)\n"
            else:
                analysis += "🔴 **PROFITABILITY:** Loss-making (ROE < 0%)\n"
            
            analysis += f"\n**Total Assets:** €{financial_data.get('total_assets', 0):,.2f}"
            analysis += f"\n**Equity:** €{financial_data.get('equity', 0):,.2f}"
            analysis += f"\n**Net Income:** €{financial_data.get('net_income', 0):,.2f}"
        
        return analysis

# ============================================================================
# VISUALIZATION
# ============================================================================

class FinancialVisualizer:
    """Create financial visualizations"""
    
    def create_metrics_chart(self, financial_data: Dict, lang: str):
        """Create bar chart of key financial metrics"""
        if not financial_data:
            return self._create_empty_chart("No data available")
        
        # Select key metrics to display
        key_metrics = ['total_assets', 'equity', 'current_assets', 'revenue', 'net_income']
        display_names = {
            'en': ['Total Assets', 'Equity', 'Current Assets', 'Revenue', 'Net Income'],
            'lv': ['Kopējie aktīvi', 'Pašu kapitāls', 'Apgrozāmie līdzekļi', 'Ieņēmumi', 'Neto ienākumi']
        }
        
        metrics = []
        values = []
        
        for metric in key_metrics:
            if metric in financial_data and financial_data[metric] != 0:
                metrics.append(display_names[lang][key_metrics.index(metric)])
                values.append(financial_data[metric])
        
        if not metrics:
            return self._create_empty_chart("No financial data")
        
        fig = go.Figure(data=[
            go.Bar(x=metrics, y=values, marker_color='#3b82f6',
                  text=[f'€{v:,.0f}' for v in values], textposition='outside')
        ])
        
        title = "Finanšu Rādītāji (EUR)" if lang == 'lv' else "Financial Metrics (EUR)"
        fig.update_layout(title=title, height=400, template="plotly_white")
        return fig
    
    def create_ratios_chart(self, ratios: Dict, lang: str):
        """Create bar chart of financial ratios"""
        if not ratios:
            return self._create_empty_chart("No ratios calculated")
        
        # Select key ratios to display
        key_ratios = ['current_ratio', 'debt_to_equity', 'return_on_equity', 'net_profit_margin']
        display_names = {
            'en': ['Current Ratio', 'Debt/Equity', 'ROE %', 'Profit Margin %'],
            'lv': ['Pašreizējais rādītājs', 'Parāds/Kapitāls', 'KP %', 'Peļņas norma %']
        }
        
        metrics = []
        values = []
        
        for ratio in key_ratios:
            if ratio in ratios:
                metrics.append(display_names[lang][key_ratios.index(ratio)])
                values.append(ratios[ratio])
        
        if not metrics:
            return self._create_empty_chart("No ratios data")
        
        fig = go.Figure(data=[
            go.Bar(x=metrics, y=values, marker_color='#10b981',
                  text=[f'{v:.2f}' for v in values], textposition='outside')
        ])
        
        title = "Finanšu Koeficienti" if lang == 'lv' else "Financial Ratios"
        fig.update_layout(title=title, height=400, template="plotly_white")
        return fig
    
    def _create_empty_chart(self, message: str):
        """Create empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, 
                          font=dict(size=16), xref="paper", yref="paper")
        fig.update_layout(height=400)
        return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class FinancialAnalysisApp:
    """Main application class"""
    
    def __init__(self):
        self.processor = LatvianFinancialReportProcessor()
        self.calculator = FinancialRatioCalculator()
        self.visualizer = FinancialVisualizer()
        self.ai_analyzer = None
    
    def set_api_key(self, api_key: str, api_provider: str):
        """Set API key for AI analysis"""
        if api_key.strip():
            self.ai_analyzer = FinancialAnalysisAI(api_key=api_key.strip(), api_provider=api_provider)
        else:
            self.ai_analyzer = FinancialAnalysisAI()
    
    def analyze_report(self, file_obj, api_key: str, api_provider: str, output_lang: str):
        """Main analysis function"""
        try:
            # Set API key
            self.set_api_key(api_key, api_provider)
            
            if file_obj is None:
                return "Please upload a PDF file", None, None, "", "No file uploaded"
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                if hasattr(file_obj, 'read'):
                    content = file_obj.read()
                else:
                    content = file_obj
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            # Extract text from PDF
            text = self.processor.extract_text_from_pdf(tmp_path)
            
            # Clean up temporary file
            import os
            os.unlink(tmp_path)
            
            if not text:
                return "Could not extract text from PDF", None, None, "", "Extraction failed"
            
            # Extract financial data
            financial_data = self.processor.extract_financial_data(text)
            company_info = self.processor.extract_company_info(text)
            
            if not financial_data:
                return "No financial data could be extracted", None, None, "", "No financial data found"
            
            # Calculate ratios
            ratios = self.calculator.calculate_all_ratios(financial_data)
            
            # Generate AI analysis
            lang = 'en' if output_lang == 'English' else 'lv'
            ai_analysis = self.ai_analyzer.analyze_financials(company_info, financial_data, ratios, lang)
            
            # Create visualizations
            metrics_chart = self.visualizer.create_metrics_chart(financial_data, lang)
            ratios_chart = self.visualizer.create_ratios_chart(ratios, lang)
            
            # Create detailed report
            report = self._create_detailed_report(company_info, financial_data, ratios, ai_analysis, lang)
            
            status = f"✅ Analysis complete | {company_info.get('company_name', 'Unknown')} | API: {api_provider.upper() if api_key else 'None'}"
            
            return report, metrics_chart, ratios_chart, ai_analysis, status
            
        except Exception as e:
            error_msg = f"Error analyzing report: {str(e)}"
            return error_msg, None, None, "", f"❌ {error_msg}"
    
    def _create_detailed_report(self, company_info: Dict, financial_data: Dict, ratios: Dict, ai_analysis: str, lang: str) -> str:
        """Create detailed financial report"""
        
        company_name = company_info.get('company_name', 'Unknown Company')
        registration = company_info.get('registration_number', 'N/A')
        period = company_info.get('period', 'N/A')
        
        if lang == 'lv':
            report = f"# 📊 Finanšu Pārskata Analīze\n## {company_name}\n"
            report += f"**Reģ. Nr.:** {registration} | **Periods:** {period}\n\n"
            
            report += "## 💰 Finanšu Dati\n"
            for key, value in financial_data.items():
                display_name = key.replace('_', ' ').title()
                report += f"- **{display_name}:** €{value:,.2f}\n"
            
            report += "\n## 📈 Finanšu Koeficienti\n"
            for key, value in ratios.items():
                display_name = key.replace('_', ' ').title()
                report += f"- **{display_name}:** {value}\n"
                
        else:
            report = f"# 📊 Financial Report Analysis\n## {company_name}\n"
            report += f"**Reg. No:** {registration} | **Period:** {period}\n\n"
            
            report += "## 💰 Financial Data\n"
            for key, value in financial_data.items():
                display_name = key.replace('_', ' ').title()
                report += f"- **{display_name}:** €{value:,.2f}\n"
            
            report += "\n## 📈 Financial Ratios\n"
            for key, value in ratios.items():
                display_name = key.replace('_', ' ').title()
                report += f"- **{display_name}:** {value}\n"
        
        report += f"\n---\n\n{ai_analysis}\n\n---\n"
        report += f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*"
        
        return report

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Create application instance
app = FinancialAnalysisApp()

# Create Gradio interface
with gr.Blocks(title="AI Financial Analyzer for Latvian Reports", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏦 AI Financial Analysis Agent
    ## Specialized for Latvian Financial Reports
    
    Upload Latvian financial reports (PDF) to extract financial data, calculate ratios, and get AI analysis.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Configuration")
            
            file_input = gr.File(
                label="Upload PDF Financial Report",
                file_types=[".pdf"],
                type="binary"
            )
            
            api_key = gr.Textbox(
                label="API Key (Optional)",
                placeholder="Enter Groq, OpenAI, or Gemini API key",
                type="password"
            )
            
            api_provider = gr.Radio(
                choices=["groq", "openai", "gemini"],
                label="API Provider",
                value="groq"
            )
            
            output_lang = gr.Radio(
                choices=["English", "Latvian"],
                label="Output Language / Izvades valoda",
                value="English"
            )
            
            analyze_btn = gr.Button("🚀 Analyze Report", variant="primary", size="lg")
            
            gr.Markdown("""
            ### ℹ️ How to use:
            1. Upload a Latvian financial report PDF
            2. (Optional) Add API key for enhanced AI analysis
            3. Select output language
            4. Click Analyze
            
            **Supported APIs:** Groq (free), OpenAI, Gemini
            """)
        
        with gr.Column(scale=2):
            status_output = gr.Textbox(label="Status", interactive=False)
            
            with gr.Tab("📋 Full Report"):
                report_output = gr.Markdown()
            
            with gr.Tab("📊 Financial Metrics"):
                metrics_chart = gr.Plot()
            
            with gr.Tab("📈 Financial Ratios"):
                ratios_chart = gr.Plot()
            
            with gr.Tab("🤖 AI Analysis"):
                ai_analysis = gr.Markdown()
    
    # Set up button click
    analyze_btn.click(
        fn=app.analyze_report,
        inputs=[file_input, api_key, api_provider, output_lang],
        outputs=[report_output, metrics_chart, ratios_chart, ai_analysis, status_output]
    )
    
    gr.Markdown("""
    ---
    ### 🎯 Features:
    - ✅ Correct extraction of Latvian number formats
    - ✅ 15+ financial ratios calculated
    - ✅ AI-powered financial analysis
    - ✅ Bilingual support (English/Latvian)
    - ✅ Multiple API support (Groq, OpenAI, Gemini)
    
    *Built specifically for Latvian financial reports*
    """)

# Launch the application
if __name__ == "__main__":
    demo.launch(share=True, debug=True)
