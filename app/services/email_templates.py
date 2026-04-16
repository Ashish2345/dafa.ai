"""
HTML email templates for MeroDafa transactional emails.

Each function returns a complete HTML string ready to be sent.
Brand: #09383e (dark teal), Georgia serif for headings.
"""


def _base(content: str) -> str:
    """Wrap content in the base email layout."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MeroDafa</title>
</head>
<body style="margin: 0; padding: 0; background-color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color: #f8fafc;">
    <tr>
      <td align="center" style="padding: 40px 16px;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width: 480px; background-color: #ffffff; border-radius: 16px; border: 1px solid #e2e8f0; overflow: hidden;">

          <!-- Logo header -->
          <tr>
            <td style="padding: 28px 32px 20px; text-align: center; border-bottom: 1px solid #f1f5f9;">
              <table role="presentation" cellpadding="0" cellspacing="0" style="margin: 0 auto;">
                <tr>
                  <td style="width: 36px; height: 36px; background-color: #09383e; border-radius: 10px; text-align: center; vertical-align: middle;">
                    <span style="color: #ffffff; font-family: Georgia, serif; font-size: 16px; font-weight: bold; line-height: 36px;">m</span>
                  </td>
                  <td style="padding-left: 10px;">
                    <span style="font-family: Georgia, serif; font-size: 20px; font-weight: bold; color: #09383e; letter-spacing: -0.3px;">merodafa</span>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <!-- Body content -->
          <tr>
            <td style="padding: 32px;">
              {content}
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="padding: 20px 32px; border-top: 1px solid #f1f5f9; text-align: center;">
              <p style="margin: 0; font-size: 12px; color: #94a3b8; line-height: 1.5;">
                MeroDafa &middot; AI Legal Research &middot; Kathmandu, Nepal
              </p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""


def verification_email(code: str, name: str, expire_minutes: int) -> str:
    """6-digit email verification template."""
    greeting = f"Hi {name}," if name else "Hi there,"
    return _base(f"""
              <p style="margin: 0 0 8px; font-size: 15px; color: #1e293b; font-weight: 600;">{greeting}</p>
              <p style="margin: 0 0 4px; font-size: 14px; color: #475569; line-height: 1.6;">
                Welcome to MeroDafa. Enter this code to verify your email and start researching:
              </p>

              <!-- OTP code box -->
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin: 24px 0;">
                <tr>
                  <td align="center">
                    <div style="display: inline-block; background-color: #f0fdf4; border: 2px solid #09383e20; border-radius: 12px; padding: 16px 32px;">
                      <span style="font-family: 'Courier New', monospace; font-size: 36px; font-weight: bold; letter-spacing: 8px; color: #09383e;">{code}</span>
                    </div>
                  </td>
                </tr>
              </table>

              <p style="margin: 0 0 4px; font-size: 13px; color: #64748b; line-height: 1.5;">
                This code expires in <strong>{expire_minutes} minutes</strong>.
              </p>
              <p style="margin: 0; font-size: 13px; color: #94a3b8; line-height: 1.5;">
                If you didn't create an account on MeroDafa, you can safely ignore this email.
              </p>
    """)


def password_reset_email(code: str, expire_minutes: int) -> str:
    """6-digit password reset template."""
    return _base(f"""
              <p style="margin: 0 0 8px; font-size: 15px; color: #1e293b; font-weight: 600;">Password reset request</p>
              <p style="margin: 0 0 4px; font-size: 14px; color: #475569; line-height: 1.6;">
                We received a request to reset your password. Enter this code to set a new one:
              </p>

              <!-- OTP code box -->
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin: 24px 0;">
                <tr>
                  <td align="center">
                    <div style="display: inline-block; background-color: #fef3c7; border: 2px solid #d9770620; border-radius: 12px; padding: 16px 32px;">
                      <span style="font-family: 'Courier New', monospace; font-size: 36px; font-weight: bold; letter-spacing: 8px; color: #92400e;">{code}</span>
                    </div>
                  </td>
                </tr>
              </table>

              <p style="margin: 0 0 4px; font-size: 13px; color: #64748b; line-height: 1.5;">
                This code expires in <strong>{expire_minutes} minutes</strong>.
              </p>
              <p style="margin: 0; font-size: 13px; color: #94a3b8; line-height: 1.5;">
                If you didn't request a password reset, ignore this email &mdash; your password will remain unchanged.
              </p>
    """)
