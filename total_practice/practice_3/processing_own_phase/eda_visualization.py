"""HTML data visualization dashboard generator for Rotten Tomatoes dataset EDA."""

from __future__ import annotations


def render_eda_dashboard() -> str:
    """Render a clean academic, self-contained, offline HTML/CSS dashboard for Rotten Tomatoes EDA."""
    return """<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; padding: 22px; border-radius: 8px; border: 1px solid #334155; margin: 16px 0; max-width: 880px;">
    <!-- Header -->
    <div style="border-bottom: 1px solid #334155; padding-bottom: 12px; margin-bottom: 18px;">
        <div style="font-size: 16px; font-weight: 700; color: #f8fafc; letter-spacing: -0.01em;">
            Rotten Tomatoes — Dataset Overview
        </div>
        <div style="font-size: 12px; color: #94a3b8; margin-top: 2px;">
            Exploratory Data Analysis
        </div>
    </div>

    <!-- Section 1: Dataset Split Overview -->
    <div style="margin-bottom: 20px;">
        <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #94a3b8; margin-bottom: 8px;">
            Dataset Split Overview
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px;">
            <div style="background: #1e293b; padding: 12px 14px; border-radius: 6px; border: 1px solid #334155; text-align: center;">
                <div style="font-size: 11px; color: #94a3b8; font-weight: 500;">Total Dataset</div>
                <div style="font-size: 22px; font-weight: 700; color: #f8fafc; margin: 4px 0;">9,596</div>
                <div style="font-size: 11px; color: #64748b; font-weight: 500;">100%</div>
            </div>
            <div style="background: #1e293b; padding: 12px 14px; border-radius: 6px; border: 1px solid #334155; text-align: center;">
                <div style="font-size: 11px; color: #94a3b8; font-weight: 500;">Train</div>
                <div style="font-size: 22px; font-weight: 700; color: #f8fafc; margin: 4px 0;">7,676</div>
                <div style="font-size: 11px; color: #38bdf8; font-weight: 500;">80%</div>
            </div>
            <div style="background: #1e293b; padding: 12px 14px; border-radius: 6px; border: 1px solid #334155; text-align: center;">
                <div style="font-size: 11px; color: #94a3b8; font-weight: 500;">Validation</div>
                <div style="font-size: 22px; font-weight: 700; color: #f8fafc; margin: 4px 0;">960</div>
                <div style="font-size: 11px; color: #c084fc; font-weight: 500;">10%</div>
            </div>
            <div style="background: #1e293b; padding: 12px 14px; border-radius: 6px; border: 1px solid #334155; text-align: center;">
                <div style="font-size: 11px; color: #94a3b8; font-weight: 500;">Holdout</div>
                <div style="font-size: 22px; font-weight: 700; color: #f8fafc; margin: 4px 0;">960</div>
                <div style="font-size: 11px; color: #facc15; font-weight: 500;">10%</div>
            </div>
        </div>
    </div>

    <!-- Section 2: Class Distribution & Text Statistics -->
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 12px; margin-bottom: 20px;">
        <!-- Class Distribution -->
        <div style="background: #1e293b; padding: 14px 16px; border-radius: 6px; border: 1px solid #334155; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #94a3b8;">
                        Class Distribution
                    </div>
                    <span style="font-size: 10px; color: #38bdf8; background: rgba(56, 189, 248, 0.1); padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(56, 189, 248, 0.2);">
                        Balanced Dataset
                    </span>
                </div>
                <!-- Negative row -->
                <div style="margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px;">
                        <span style="color: #f87171; font-weight: 600;">Negative</span>
                        <span style="color: #f8fafc; font-weight: 600;">4,798 &nbsp;<span style="color: #94a3b8; font-weight: 400;">(50.0%)</span></span>
                    </div>
                    <div style="background: #0f172a; height: 6px; border-radius: 3px; overflow: hidden; border: 1px solid #334155;">
                        <div style="background: #f87171; width: 50%; height: 100%;"></div>
                    </div>
                </div>
                <!-- Positive row -->
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px;">
                        <span style="color: #4ade80; font-weight: 600;">Positive</span>
                        <span style="color: #f8fafc; font-weight: 600;">4,798 &nbsp;<span style="color: #94a3b8; font-weight: 400;">(50.0%)</span></span>
                    </div>
                    <div style="background: #0f172a; height: 6px; border-radius: 3px; overflow: hidden; border: 1px solid #334155;">
                        <div style="background: #4ade80; width: 50%; height: 100%;"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Text Statistics Table -->
        <div style="background: #1e293b; padding: 14px 16px; border-radius: 6px; border: 1px solid #334155; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #94a3b8; margin-bottom: 8px;">
                    Text Statistics
                </div>
                <table style="width: 100%; border-collapse: collapse; font-size: 12px; text-align: right;">
                    <thead>
                        <tr style="border-bottom: 1px solid #334155; color: #94a3b8; font-size: 11px;">
                            <th style="text-align: left; padding: 4px 0; font-weight: 500;">Metric</th>
                            <th style="padding: 4px 6px; font-weight: 500;">Mean</th>
                            <th style="padding: 4px 6px; font-weight: 500;">Median</th>
                            <th style="padding: 4px 6px; font-weight: 500;">Min</th>
                            <th style="padding: 4px 0; font-weight: 500;">Max</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom: 1px solid rgba(51, 65, 85, 0.5);">
                            <td style="text-align: left; padding: 6px 0; color: #f8fafc; font-weight: 500;">Characters</td>
                            <td style="padding: 6px 6px; color: #f8fafc;">114.0</td>
                            <td style="padding: 6px 6px; color: #94a3b8;">111.0</td>
                            <td style="padding: 6px 6px; color: #64748b;">4</td>
                            <td style="padding: 6px 0; color: #64748b;">267</td>
                        </tr>
                        <tr>
                            <td style="text-align: left; padding: 6px 0; color: #f8fafc; font-weight: 500;">Words</td>
                            <td style="padding: 6px 6px; color: #f8fafc;">21.0</td>
                            <td style="padding: 6px 6px; color: #94a3b8;">20.0</td>
                            <td style="padding: 6px 6px; color: #64748b;">1</td>
                            <td style="padding: 6px 0; color: #64748b;">59</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            <div style="font-size: 11px; color: #94a3b8; margin-top: 8px; border-top: 1px solid #334155; padding-top: 6px; display: flex; justify-content: space-between;">
                <span>Maximum token length: <strong style="color: #f8fafc;">80</strong></span>
                <span>Token coverage: <strong style="color: #38bdf8;">100%</strong></span>
            </div>
        </div>
    </div>

    <!-- Section 3: Sample Reviews -->
    <div>
        <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #94a3b8; margin-bottom: 8px;">
            Sample Reviews
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 10px;">
            <!-- Negative Sample -->
            <div style="background: #1e293b; border-left: 3px solid #f87171; padding: 12px 14px; border-radius: 0 6px 6px 0; border-top: 1px solid #334155; border-right: 1px solid #334155; border-bottom: 1px solid #334155; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="color: #f87171; font-weight: 600; font-size: 11px;">Negative</span>
                        <span style="font-size: 10px; color: #64748b;">Class 0</span>
                    </div>
                    <div style="font-size: 13px; color: #f8fafc; line-height: 1.45; font-style: italic;">
                        &ldquo;simplistic , silly and tedious .&rdquo;
                    </div>
                </div>
                <div style="font-size: 10px; color: #64748b; margin-top: 8px;">
                    32 characters · 6 words
                </div>
            </div>

            <!-- Positive Sample -->
            <div style="background: #1e293b; border-left: 3px solid #4ade80; padding: 12px 14px; border-radius: 0 6px 6px 0; border-top: 1px solid #334155; border-right: 1px solid #334155; border-bottom: 1px solid #334155; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="color: #4ade80; font-weight: 600; font-size: 11px;">Positive</span>
                        <span style="font-size: 10px; color: #64748b;">Class 1</span>
                    </div>
                    <div style="font-size: 13px; color: #f8fafc; line-height: 1.45; font-style: italic;">
                        &ldquo;the rock is destined to be the 21st century&#39;s new &quot; conan &quot; and that he&#39;s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .&rdquo;
                    </div>
                </div>
                <div style="font-size: 10px; color: #64748b; margin-top: 8px;">
                    177 characters · 34 words
                </div>
            </div>
        </div>
    </div>
</div>"""
